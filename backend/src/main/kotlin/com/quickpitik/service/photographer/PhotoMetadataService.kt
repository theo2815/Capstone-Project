package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PhotoMetadataProperties
import com.quickpitik.dto.photos.PhotoMetadataDto
import com.quickpitik.dto.photos.toDto
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.PhotoMetadataRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.image.PhotoMetadataExtractor
import com.quickpitik.service.image.hasRecordedTags
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.RejectedExecutionException

@Service
class PhotoMetadataService(
    private val repository: PhotoMetadataRepository,
    private val photos: PhotoRepository,
    private val coverage: EventPhotographerRepository,
    private val extractor: PhotoMetadataExtractor,
    private val storage: StorageService,
    @Qualifier("watermarkProcessing") private val watermarkPool: ThreadPoolTaskExecutor,
    private val properties: PhotoMetadataProperties,
    private val transactions: TransactionTemplate,
    private val clock: Clock,
    @Qualifier("metadataProcessing") private val metadataPool: ThreadPoolTaskExecutor,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional(readOnly = true)
    fun getOwned(userId: UUID, photoId: UUID): PhotoMetadataDto {
        photos.findByIdAndPhotographerIdAndDeletionStartedAtIsNull(photoId, userId) ?: throw notFound()
        return repository.findById(photoId).orElse(null).toDto(photoId)
    }

    @Transactional(readOnly = true)
    fun getPublic(photoId: UUID): PhotoMetadataDto {
        val photo = photos.findById(photoId).orElse(null)
        if (photo == null || photo.status != PhotoStatus.LIVE || photo.deletedAt != null || photo.deletionStartedAt != null ||
            metadataAccess(photo.eventId, listOf(photo), null)[photo.id] != "public") {
            throw notFound()
        }
        return repository.findById(photoId).orElse(null).toDto(photoId)
    }

    @Transactional(readOnly = true)
    fun metadataAccess(eventId: UUID, photos: Collection<Photo>, requesterUserId: UUID?): Map<UUID, String> {
        val photographerIds = photos.mapNotNullTo(mutableSetOf()) { it.photographerId }
        val publicIds = if (photographerIds.isEmpty()) emptySet()
        else coverage.findPublicMetadataPhotographerIds(eventId, photographerIds).toSet()
        return photos.associate { photo ->
            photo.id to when {
                requesterUserId != null && photo.photographerId == requesterUserId -> "owner"
                photo.photographerId in publicIds -> "public"
                else -> "none"
            }
        }
    }

    private fun notFound() = NotFoundException(code = ErrorCodes.NOT_FOUND, message = "Photo not found")

    fun dispatchExtraction(photoId: UUID, originalBytes: ByteArray) {
        // Enforce the retained-byte ceiling even for legacy/oversized originals.
        // No claim before submission: rejection/shutdown leaves PENDING recoverable.
        if (originalBytes.size > MAX_ORIGINAL_BYTES) return
        try {
            metadataPool.execute { extractIfDue(photoId, originalBytes) }
        } catch (_: RejectedExecutionException) {
            log.debug("Metadata submission deferred for photo {}", photoId)
        }
    }

    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    fun extractIfDue(photoId: UUID, originalBytes: ByteArray) {
        try {
            val token = UUID.randomUUID()
            val now = OffsetDateTime.now(clock)
            val claimed = transactions.execute {
                repository.claimIfDue(photoId, token, EXTRACTOR_VERSION, now, now.plusMinutes(LEASE_MINUTES), MAX_ATTEMPTS)
            }
            if (claimed != 1) return
            extractClaimed(photoId, token, originalBytes)
        } catch (ex: Exception) {
            log.warn("Metadata persistence failed for photo {} ({})", photoId, ex.javaClass.simpleName)
        }
    }

    @Scheduled(fixedDelayString = "\${app.photo-metadata.recovery-interval-ms:60000}")
    fun recoverDue() {
        if (!properties.recoveryEnabled || watermarkPool.activeCount != 0 ||
            watermarkPool.threadPoolExecutor.queue.isNotEmpty() || metadataPool.activeCount != 0 ||
            metadataPool.threadPoolExecutor.queue.isNotEmpty()) return
        val capacity = minOf(RECOVERY_BATCH_SIZE, metadataPool.maxPoolSize - metadataPool.activeCount)
        if (capacity <= 0) return
        val now = OffsetDateTime.now(clock)
        repository.findDueCandidateIds(EXTRACTOR_VERSION, MAX_ATTEMPTS, now.minusMinutes(PENDING_GRACE_MINUTES), now,
            PageRequest.of(0, capacity)).forEach { photoId ->
            try {
                metadataPool.execute { recoverOne(photoId) }
            } catch (_: RejectedExecutionException) {
                log.debug("Metadata recovery deferred for photo {}", photoId)
            }
        }
    }

    private fun recoverOne(photoId: UUID) {
        try {
            val token = UUID.randomUUID()
            val now = OffsetDateTime.now(clock)
            val claimed = transactions.execute {
                repository.claimIfDue(photoId, token, EXTRACTOR_VERSION, now, now.plusMinutes(LEASE_MINUTES), MAX_ATTEMPTS)
            }
            if (claimed != 1) {
                transactions.executeWithoutResult {
                    repository.failExpiredExhaustedLease(photoId, EXTRACTOR_VERSION, MAX_ATTEMPTS, now)
                }
                return
            }
            val key = try { photos.findById(photoId).orElse(null)?.s3Key } catch (ex: Exception) {
                settleStorageFailure(photoId, token)
                return
            } ?: return
            val head = try { storage.metadata(key) } catch (ex: Exception) {
                settleStorageFailure(photoId, token)
                return
            }
            if (head == null || head.sizeBytes > MAX_ORIGINAL_BYTES) {
                settleUnavailable(photoId, token, head?.sizeBytes)
                return
            }
            val bytes = try {
                storage.open(key).use { it.readNBytes((MAX_ORIGINAL_BYTES + 1).toInt()) }
            } catch (ex: Exception) {
                val disappeared = try { storage.metadata(key) == null } catch (_: Exception) { false }
                if (disappeared) settleUnavailable(photoId, token, null)
                else settleStorageFailure(photoId, token)
                return
            }
            when {
                bytes.size > MAX_ORIGINAL_BYTES -> settleUnavailable(photoId, token, bytes.size.toLong())
                bytes.size.toLong() != head.sizeBytes -> settleStorageFailure(photoId, token)
                else -> extractClaimed(photoId, token, bytes, head.sizeBytes)
            }
        } catch (ex: Exception) {
            log.warn("Metadata recovery failed for photo {} ({})", photoId, ex.javaClass.simpleName)
        }
    }

    private fun settleStorageFailure(photoId: UUID, token: UUID) {
        val now = OffsetDateTime.now(clock)
        transactions.executeWithoutResult {
            repository.settleStorageFailure(photoId, token, EXTRACTOR_VERSION, now,
                now.plusMinutes(1), now.plusMinutes(5), now.plusMinutes(30))
        }
    }

    private fun settleUnavailable(photoId: UUID, token: UUID, sizeBytes: Long?) {
        transactions.executeWithoutResult {
            repository.settleUnavailable(photoId, token, EXTRACTOR_VERSION, sizeBytes, OffsetDateTime.now(clock))
        }
    }

    private fun extractClaimed(photoId: UUID, token: UUID, originalBytes: ByteArray, verifiedSizeBytes: Long? = null) {
        // Only parsing failures are terminal. A database outage leaves the
        // committed lease for recovery instead of mislabelling the bytes.
        val fields = try {
            originalBytes.inputStream().use(extractor::extract)
        } catch (ex: Exception) {
            log.warn("Metadata parsing failed for photo {} ({})", photoId, ex.javaClass.simpleName)
            null
        }
        val state = when {
            fields == null -> PhotoMetadataState.FAILED
            fields.hasRecordedTags() -> PhotoMetadataState.READY
            else -> PhotoMetadataState.EMPTY
        }
        transactions.executeWithoutResult {
            repository.settle(photoId, token, EXTRACTOR_VERSION, state, fields, OffsetDateTime.now(clock), verifiedSizeBytes)
        }
    }

    private companion object {
        const val EXTRACTOR_VERSION = 1
        const val LEASE_MINUTES = 5L
        const val MAX_ATTEMPTS = 4
        const val PENDING_GRACE_MINUTES = 5L
        const val MAX_ORIGINAL_BYTES = 25L * 1024 * 1024
        const val RECOVERY_BATCH_SIZE = 10
    }
}
