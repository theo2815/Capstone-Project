package com.quickpitik.service.removal

import com.fasterxml.jackson.databind.ObjectMapper
import com.drew.imaging.FileType
import com.drew.imaging.FileTypeDetector
import com.drew.metadata.Metadata
import com.drew.metadata.webp.WebpDirectory
import com.drew.metadata.webp.WebpRiffHandler
import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.image.ImagePixelGuard
import com.quickpitik.service.profile.SelfieService
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.MeterRegistry
import org.springframework.data.domain.PageRequest
import org.springframework.http.HttpStatus
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.Clock
import java.util.UUID
import javax.imageio.ImageIO

enum class EvidenceCleanupResult { DELETED, RETRY_REQUIRED }
class RemovalEvidenceStream(val stream: InputStream, val contentType: String,
    val eventId: UUID, val requestVersion: Long, val evidenceGeneration: Long, val evidenceKey: String)

@Service
@Transactional(propagation = Propagation.NEVER)
class PhotoRemovalEvidenceService(
    private val requests: PhotoRemovalRequestRepository,
    private val commands: PhotoRemovalCommandRepository,
    private val items: PhotoRemovalItemRepository,
    private val storage: StorageService,
    private val reviews: PhotoRemovalReviewRepository,
    private val history: PhotoRemovalHistoryRecorder,
    private val historyReader: PhotoRemovalHistoryRepository,
    private val transitions: PhotoRemovalTransitions,
    private val tx: TransactionTemplate,
    private val mapper: ObjectMapper,
    private val clock: Clock,
    private val properties: PhotoRemovalProperties,
    private val metrics: MeterRegistry,
    private val users: UserRepository,
) {
    fun replace(adminId: UUID, requestId: UUID, commandId: UUID, expectedVersion: Long, image: SearchImage): PhotoRemovalRequestDto {
        val hash = commandHash(mapper, listOf(expectedVersion.toString(), image.contentType, image.filename, hashBytes(image.bytes)))
        val key = "photo-removal/$requestId/evidence/$commandId"
        val replay = tx.execute {
            requireActiveAdmin(users, adminId)
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            commands.findByCommandId(commandId)?.let { return@execute it.replay(adminId, requestId, RemovalCommandOperation.REPLACE_EVIDENCE, hash, mapper) }
            transitions.ensureVersion(request, expectedVersion)
            if (request.status !in setOf(RemovalRequestStatus.DRAFT, RemovalRequestStatus.OPEN) ||
                commands.findByRequestIdAndStateIn(requestId, listOf(RemovalCommandState.PENDING)).isNotEmpty()) removalConflict()
            null
        }
        if (replay != null) return replay
        // Header inspection and decoding must happen before a durable write claim, outside a transaction.
        validateImage(image)
        val claimedReplay = tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            commands.findByCommandId(commandId)?.let {
                return@execute it.replay(adminId, requestId, RemovalCommandOperation.REPLACE_EVIDENCE, hash, mapper)
            }
            transitions.ensureVersion(request, expectedVersion)
            if (request.status !in setOf(RemovalRequestStatus.DRAFT, RemovalRequestStatus.OPEN) ||
                commands.findByRequestIdAndStateIn(requestId, listOf(RemovalCommandState.PENDING)).isNotEmpty()) removalConflict()
            if (commands.insertClaim(commandId, requestId, RemovalCommandOperation.REPLACE_EVIDENCE.name,
                    adminId, hash, key, clock.instant()) == 0) {
                // ON CONFLICT leaves the transaction usable so the winning identity can be checked.
                return@execute commands.findByCommandId(commandId)!!.replay(adminId, requestId, RemovalCommandOperation.REPLACE_EVIDENCE, hash, mapper)
            }
            request.outstandingEvidenceKeys = (request.outstandingEvidenceKeys + key).distinct()
            request.evidenceCleanupState = RemovalEvidenceCleanupState.PENDING
            request.evidenceCleanupNextAttemptAt = clock.instant().plus(properties.cleanupBackoff)
            request.updatedAt = clock.instant()
            requests.flush()
            null
        }
        if (claimedReplay != null) return claimedReplay
        try {
            storage.put(key, image.bytes, image.contentType.lowercase())
        } catch (error: Exception) {
            tx.executeWithoutResult {
                val request = requests.findForUpdate(requestId) ?: removalNotFound()
                val receipt = commands.findByCommandId(commandId) ?: removalConflict()
                receipt.ensureIdentity(adminId, requestId, RemovalCommandOperation.REPLACE_EVIDENCE, hash)
                if (receipt.intendedEvidenceKey != key || receipt.state != RemovalCommandState.PENDING) removalConflict()
                receipt.state = RemovalCommandState.UNKNOWN
                request.evidenceCleanupState = RemovalEvidenceCleanupState.RETRY_REQUIRED
                request.evidenceCleanupNextAttemptAt = clock.instant().plus(properties.cleanupBackoff)
                requests.flush()
            }
            outcome("write_unresolved")
            throw unavailable()
        }
        val result = tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            val receipt = commands.findByCommandId(commandId) ?: removalConflict()
            receipt.ensureIdentity(adminId, requestId, RemovalCommandOperation.REPLACE_EVIDENCE, hash)
            if (receipt.intendedEvidenceKey != key || receipt.state != RemovalCommandState.PENDING) removalConflict()
            if (request.status !in setOf(RemovalRequestStatus.DRAFT, RemovalRequestStatus.OPEN)) {
                // PUT completed after cancellation: bytes are known settled, but never install them as evidence.
                receipt.state = RemovalCommandState.FAILED
                request.evidenceCleanupNextAttemptAt = clock.instant()
                return@execute null
            }
            request.evidenceKey?.let { request.outstandingEvidenceKeys = (request.outstandingEvidenceKeys + it).distinct() }
            request.outstandingEvidenceKeys = request.outstandingEvidenceKeys - key
            request.evidenceKey = key
            request.evidenceContentType = image.contentType.lowercase()
            request.evidenceGeneration++
            request.candidateCache = null
            request.updatedAt = clock.instant()
            request.evidenceCleanupState = if (request.outstandingEvidenceKeys.isEmpty()) RemovalEvidenceCleanupState.PRESENT else RemovalEvidenceCleanupState.PENDING
            request.evidenceCleanupNextAttemptAt = if (request.outstandingEvidenceKeys.isEmpty()) null else clock.instant()
            // Candidates belong to the earlier evidence generation. Preserve verification and owner deadlines.
            items.findByRequestIdOrderByIdAsc(requestId).filter { it.state == RemovalItemState.CANDIDATE }
                .forEach { it.state = RemovalItemState.EXCLUDED }
            items.flush()
            requests.flush()
            history.record(RemovalHistoryEvent(requestId = requestId, actorType = RemovalActorType.ADMIN, actorId = adminId,
                action = RemovalHistoryActions.EVIDENCE_REPLACED, occurredAt = clock.instant(), commandId = commandId,
                metadata = mapOf("generation" to request.evidenceGeneration.toString())))
            val result = request.toAdminDto(items.findByRequestIdOrderByIdAsc(requestId), reviews.findByRequestIdOrderByIdAsc(requestId))
            receipt.result = mapper.valueToTree(result)
            receipt.state = RemovalCommandState.SUCCEEDED
            commands.flush()
            result
        }
        if (result == null) {
            purgeResolvedEvidence(requestId)
            removalConflict()
        }
        outcome("write_confirmed")
        return result
    }

    fun stream(adminId: UUID, requestId: UUID): RemovalEvidenceStream {
        val evidence = tx.execute {
            requireActiveAdmin(users, adminId)
            val request = requests.findById(requestId).orElse(null) ?: removalNotFound()
            EvidenceSnapshot(request.evidenceKey ?: removalNotFound(), request.evidenceContentType ?: removalNotFound(),
                request.eventIdSnapshot, request.version, request.evidenceGeneration)
        }!!
        return try { RemovalEvidenceStream(storage.open(evidence.key), evidence.contentType, evidence.eventId,
            evidence.version, evidence.generation, evidence.key) } catch (error: Exception) { throw unavailable() }
    }

    private data class EvidenceSnapshot(val key: String, val contentType: String, val eventId: UUID, val version: Long, val generation: Long)

    fun purgeResolvedEvidence(requestId: UUID): EvidenceCleanupResult {
        val work = tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            val terminal = request.status in setOf(RemovalRequestStatus.RESOLVING, RemovalRequestStatus.RESOLVED, RemovalRequestStatus.CANCELLED)
            if (terminal) request.candidateCache = null
            val unresolved = commands.findByRequestIdAndStateIn(requestId, listOf(RemovalCommandState.PENDING, RemovalCommandState.UNKNOWN))
                .mapNotNull { it.intendedEvidenceKey }.toSet()
            val keys = (request.outstandingEvidenceKeys + if (terminal) listOfNotNull(request.evidenceKey) else emptyList()).distinct()
            request.evidenceCleanupNextAttemptAt = clock.instant().plus(properties.cleanupBackoff)
            (keys - unresolved) to unresolved
        }!!
        val deleted = work.first.filter { key -> try { storage.delete(key); !storage.exists(key) } catch (error: Exception) { false } }.toSet()
        val result = tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            request.outstandingEvidenceKeys = request.outstandingEvidenceKeys.filterNot { it in deleted }
            if (request.evidenceKey in deleted) {
                request.evidenceKey = null
                request.evidenceContentType = null
            }
            val unresolved = commands.findByRequestIdAndStateIn(requestId, listOf(RemovalCommandState.PENDING, RemovalCommandState.UNKNOWN)).isNotEmpty()
            val terminal = request.status in setOf(RemovalRequestStatus.RESOLVING, RemovalRequestStatus.RESOLVED, RemovalRequestStatus.CANCELLED)
            val pending = unresolved || request.outstandingEvidenceKeys.isNotEmpty() || (terminal && request.evidenceKey != null)
            request.evidenceCleanupState = when {
                pending -> RemovalEvidenceCleanupState.RETRY_REQUIRED
                request.evidenceKey != null -> RemovalEvidenceCleanupState.PRESENT
                else -> RemovalEvidenceCleanupState.DELETED
            }
            request.evidenceCleanupNextAttemptAt = if (pending) clock.instant().plus(properties.cleanupBackoff) else null
            request.updatedAt = clock.instant()
            if (!pending && terminal && request.resolvedAt == null) {
                if (!historyReader.existsByRequestIdAndAction(requestId, RemovalHistoryActions.EVIDENCE_PURGED)) {
                    history.record(RemovalHistoryEvent(requestId = requestId, actorType = RemovalActorType.SYSTEM,
                        action = RemovalHistoryActions.EVIDENCE_PURGED, occurredAt = clock.instant(),
                        commandId = cleanupCommand(requestId, "purged")))
                }
                // TASK 11 owns other terminal resolution. Only an audited cancellation closes here.
                if (request.status == RemovalRequestStatus.RESOLVING && historyReader.existsByRequestIdAndAction(requestId, RemovalHistoryActions.CANCEL_REQUESTED)) {
                    request.status = RemovalRequestStatus.CANCELLED
                    request.resolvedAt = clock.instant()
                    history.record(RemovalHistoryEvent(requestId = requestId, actorType = RemovalActorType.SYSTEM,
                        action = RemovalHistoryActions.CANCELLED, fromState = "RESOLVING", toState = "CANCELLED",
                        occurredAt = request.resolvedAt!!, commandId = cleanupCommand(requestId, "cancelled")))
                }
            }
            requests.flush()
            if (request.status in setOf(RemovalRequestStatus.RESOLVED, RemovalRequestStatus.CANCELLED)) commands.sanitizeTerminalResults(requestId)
            if (pending) EvidenceCleanupResult.RETRY_REQUIRED else EvidenceCleanupResult.DELETED
        }!!
        outcome(if (result == EvidenceCleanupResult.DELETED) "deleted" else "retry_required")
        return result
    }

    @Scheduled(fixedDelayString = "\${app.photo-removal.cleanup-interval:PT1M}", initialDelayString = "\${app.photo-removal.cleanup-interval:PT1M}")
    fun retryEvidenceCleanup() {
        requests.findByEvidenceCleanupNextAttemptAtLessThanEqualOrderByEvidenceCleanupNextAttemptAtAscIdAsc(clock.instant(), PageRequest.of(0, 50))
            .forEach { try { purgeResolvedEvidence(it.id) } catch (error: Exception) { outcome("settle_failed") } }
    }

    private fun validateImage(image: SearchImage) {
        if (image.bytes.isEmpty()) removalInvalid("Selfie is empty")
        if (image.contentType.lowercase() !in SelfieService.SUPPORTED_TYPES) removalInvalid("Selfie must be jpeg, png, or webp")
        if (image.bytes.size > SelfieService.MAX_SELFIE_BYTES) {
            throw ApiException(HttpStatus.PAYLOAD_TOO_LARGE, ErrorCodes.PAYLOAD_TOO_LARGE, "Selfie exceeds the supported size", "selfie")
        }
        if (image.contentType.equals("image/webp", ignoreCase = true)) {
            validateWebp(image.bytes)
            return
        }
        if (ImagePixelGuard.exceedsPixelBudget(image.bytes)) removalInvalid("Image dimensions exceed the supported maximum")
        val decoded = try { ImageIO.read(image.bytes.inputStream()) } catch (error: Exception) { null }
        if (decoded == null) removalInvalid("Selfie cannot be decoded")
    }

    private fun validateWebp(bytes: ByteArray) {
        // Bound the container and every frame/chunk before allocating any metadata payload.
        if (bytes.size < 12) removalInvalid("Selfie is not a valid WebP image")
        val header = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        fun unsignedInt(offset: Int) = header.getInt(offset).toLong() and 0xffffffffL
        if (unsignedInt(4) != bytes.size.toLong() - 8) removalInvalid("Selfie is not a valid WebP image")
        val type = try { FileTypeDetector.detectFileType(bytes.inputStream()) }
            catch (error: Exception) { removalInvalid("Selfie is not a valid WebP image") }
        if (type != FileType.WebP) removalInvalid("Selfie is not a valid WebP image")
        fun pixelBudget(width: Int, height: Int) {
            if (width <= 0 || height <= 0) removalInvalid("WebP dimensions are invalid")
            if (width.toLong() * height > ImagePixelGuard.MAX_PIXELS) removalInvalid("Image dimensions exceed the supported maximum")
        }
        fun dimensionChunk(name: String, start: Int, end: Int) {
            // One bounded metadata object at a time, including individual animated frame images.
            val metadata = Metadata()
            try { WebpRiffHandler(metadata).processChunk(name, bytes.copyOfRange(start, end)) }
            catch (error: Exception) { removalInvalid("Selfie is not a valid WebP image") }
            val dimensions = metadata.getFirstDirectoryOfType(WebpDirectory::class.java)
                ?: removalInvalid("WebP image metadata is missing")
            if (metadata.hasErrors()) removalInvalid("WebP image metadata is invalid")
            pixelBudget(dimensions.getInteger(WebpDirectory.TAG_IMAGE_WIDTH) ?: removalInvalid("WebP dimensions are missing"),
                dimensions.getInteger(WebpDirectory.TAG_IMAGE_HEIGHT) ?: removalInvalid("WebP dimensions are missing"))
        }
        fun unsigned24(offset: Int) = (bytes[offset].toInt() and 255) or
            ((bytes[offset + 1].toInt() and 255) shl 8) or ((bytes[offset + 2].toInt() and 255) shl 16)
        fun chunks(start: Int, limit: Int, frame: Boolean, readMetadata: Boolean): Boolean {
            var offset = start
            var hasImage = false
            while (offset < limit) {
                if (limit - offset < 8) removalInvalid("WebP chunk is truncated")
                val size = unsignedInt(offset + 4)
                val body = offset + 8
                val end = body.toLong() + size + (size and 1)
                if (end > limit) removalInvalid("WebP chunk is truncated")
                val name = String(bytes, offset, 4, Charsets.US_ASCII)
                when (name) {
                    "VP8 ", "VP8L" -> {
                        if (size <= if (name == "VP8L") 5 else 10) removalInvalid("WebP image payload is missing")
                        if (readMetadata) dimensionChunk(name, body, (body + size).toInt())
                        hasImage = true
                    }
                    "VP8X" -> {
                        if (frame || size != 10L) removalInvalid("WebP canvas is invalid")
                        if (readMetadata) dimensionChunk(name, body, body + 10)
                    }
                    "ANMF" -> {
                        if (frame || size < 16) removalInvalid("WebP frame is truncated")
                        pixelBudget(unsigned24(body + 6) + 1, unsigned24(body + 9) + 1)
                        if (!chunks(body + 16, (body + size).toInt(), true, readMetadata)) removalInvalid("WebP frame image is missing")
                        hasImage = true
                    }
                    // Optional and unknown chunks, including empty chunks, stay in the original bytes.
                }
                offset = end.toInt()
            }
            return hasImage
        }
        if (!chunks(12, bytes.size, false, false)) removalInvalid("WebP image payload is missing")
        chunks(12, bytes.size, false, true)
    }

    private fun cleanupCommand(requestId: UUID, action: String): UUID = UUID.nameUUIDFromBytes("$requestId:$action".toByteArray())
    private fun outcome(result: String) { metrics.counter("qp.removal.evidence", "outcome", result).increment() }
    private fun unavailable() = ApiException(HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.UPLOAD_STORAGE_UNAVAILABLE, "Case evidence storage is unavailable")
}
