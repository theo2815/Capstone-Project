package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.DirectUploadBeginResponse
import com.quickpitik.dto.photographer.PhotoExistsResponse
import com.quickpitik.dto.photographer.PhotoExistsResult
import com.quickpitik.dto.photographer.UploadedPhotoDto
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoMetadata
import com.quickpitik.entity.PhotoSpan
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotoUploadIntent
import com.quickpitik.entity.PhotoUploadIntentState
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoUploadIntentRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.service.image.PhotoUploadValidator
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.MeterRegistry
import io.micrometer.core.instrument.Timer
import jakarta.persistence.EntityManager
import org.hibernate.exception.ConstraintViolationException
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.data.domain.PageRequest
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
import org.springframework.web.multipart.MultipartFile
import java.io.ByteArrayInputStream
import java.text.Normalizer
import java.time.Clock
import java.time.Duration
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.UUID
import java.util.concurrent.Semaphore
import java.util.concurrent.atomic.AtomicLong
import kotlin.random.Random

@Service
class PhotoUploadService(
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val photoRepository: PhotoRepository,
    private val photoUploadIntentRepository: PhotoUploadIntentRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val eventRepository: EventRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val accessGuard: PhotographerAccessGuard,
    private val photoUploadValidator: PhotoUploadValidator,
    private val aiApiProperties: AiApiProperties,
    private val eventPublisher: ApplicationEventPublisher,
    private val transactionTemplate: TransactionTemplate,
    private val meterRegistry: MeterRegistry,
    private val entityManager: EntityManager,
    private val clock: Clock,
) {
    private val log = LoggerFactory.getLogger(javaClass)
    private val verificationSlots = Semaphore(storageProperties.directUpload.verificationConcurrency)
    private val pendingCount = meterRegistry.gauge("qp.upload.direct.pending.count", AtomicLong())!!
    private val pendingBytes = meterRegistry.gauge("qp.upload.direct.pending.bytes", AtomicLong())!!
    private val oldestPendingAge = meterRegistry.gauge("qp.upload.direct.pending.oldest.age", AtomicLong())!!

    // Deliberately NOT @Transactional: the original-object storage PUT below
    // takes long enough that it would pin a Hikari connection (pool of 10) for
    // the ride — marathon-day concurrent uploads exhausted the pool that way.
    // Only the short persist block at the end runs in a transaction
    // (TransactionTemplate). The watermark decode + composite + PUT moved off
    // this request entirely (PhotoWatermarkService, 2026-08-28).
    fun upload(photographerId: UUID, eventId: UUID, file: MultipartFile): UploadedPhotoDto {
        // Times SUCCESSFUL new-photo uploads only (dedup short-circuits and
        // validation rejects return before the stop) — the number that matters
        // for marathon-day capacity planning.
        val timerSample = Timer.start(meterRegistry)
        val (event, settings) = gate(photographerId, eventId)

        val contentType = file.contentType?.lowercase()
        if (file.isEmpty || contentType == null || contentType !in ALLOWED_CONTENT_TYPES) {
            throw ApiException(
                status = HttpStatus.UNSUPPORTED_MEDIA_TYPE,
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "Upload must be image/jpeg, image/png, or image/webp.",
            )
        }

        val bytes = file.bytes
        val contentHash = photoUploadValidator.validate(
            ByteArrayInputStream(bytes),
            contentType,
            bytes.size.toLong(),
        )

        // Duplicate detection (enterprise dedup). A photo's identity is the
        // SHA-256 of its ORIGINAL bytes, hashed HERE — before watermarking,
        // which would change them — so a client- and server-side hash of the
        // same file agree. Boundary is per-photographer across all events: the
        // same shot can't be uploaded twice. A same-event re-upload (the
        // stop-on-mobile / continue-on-web case) returns the existing photo
        // idempotently; a different-event re-upload is rejected. The partial
        // unique index (photographer_id, content_hash) from V24 is the
        // race-safe backstop (see persistNew below).
        duplicateOf(photographerId, eventId, contentHash)?.let { return existingPhotoDto(it) }

        // Blur culling is desktop-only (BatchMyPhotos). Web upload assumes
        // photos have already been culled by the photographer's desktop
        // workflow — see backend/decisions 2026-05-18 "Blur removed from
        // BE upload" + website/decisions 2026-05-06 "Blur removed from web."

        val photoId = UUID.randomUUID()
        val originalKey = originalKeyFor(eventId, photoId)

        // Watermark generation moved OFF this request (2026-08-28): the decode +
        // composite (~200-500ms CPU) and the watermark PUT now run async in
        // PhotoWatermarkService, which flips the photo PROCESSING → LIVE when
        // the derivative lands. The upload's 200 guarantees only that the
        // ORIGINAL is durably stored (the mobile worker deletes its local copy
        // on 200, so that invariant is load-bearing). A cheap pointer check
        // keeps the actionable WATERMARK_MISSING 422 for the common
        // misconfiguration; the storage-drift case (pointer present, bytes
        // gone) now surfaces async and the sweep retries it after the
        // photographer re-uploads the logo in Settings.
        requireWatermark(settings, event)

        storageService.put(originalKey, bytes, contentType)

        val dto = persistNew(photoId, eventId, photographerId, originalKey, contentHash, event,
            originalFilename = file.originalFilename, originalSizeBytes = bytes.size.toLong())
        timerSample.stop(meterRegistry.timer("qp.upload.duration"))
        return dto
    }

    // ── Direct-to-storage upload (2026-09-02) ───────────────────────────────
    // The multipart path pushes every byte through this server twice (in from
    // the phone, out to storage) — measured at ~6 s per 1.2 MB frame on the
    // 2026-09-02 device session, all of it the storage PUT. These two steps let
    // the client PUT straight to S3/R2 with a presigned URL and then register
    // the object. Commit still performs one bounded server-side read so the
    // gates, byte validation, dedup and persist match upload().
    //
    // The client supplies the expected hash, but commit snapshots the staging
    // object and verifies its actual bytes before any marketplace row exists.

    fun beginDirectUpload(
        photographerId: UUID,
        eventId: UUID,
        contentHash: String,
        contentType: String,
        sizeBytes: Long,
        requestId: UUID? = null,
        client: String? = null,
        originalFilename: String? = null,
    ): DirectUploadBeginResponse {
        if (client != null && client != WEBSITE_CLIENT) {
            throw ApiException(HttpStatus.BAD_REQUEST, ErrorCodes.VALIDATION_ERROR, "Unknown upload client.")
        }
        val type = contentType.lowercase()
        val hash = contentHash.lowercase()
        if (type !in ALLOWED_CONTENT_TYPES || sizeBytes !in 1..storageProperties.directUpload.maxBytes) {
            throw ApiException(
                status = HttpStatus.UNSUPPORTED_MEDIA_TYPE,
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "Upload must be image/jpeg, image/png, or image/webp under 25 MB.",
            )
        }
        if (!photoUploadValidator.canDecode(type)) {
            throw ApiException(
                HttpStatus.UNSUPPORTED_MEDIA_TYPE,
                ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                "That image type is not supported by the server image decoder.",
            )
        }

        if (requestId != null) {
            accessGuard.requireApproved(photographerId, "uploading")
            photoUploadIntentRepository.findByPhotographerIdAndRequestId(photographerId, requestId)?.let {
                validateReplay(it, eventId, hash, type, sizeBytes)
                return replayBegin(it)
            }
        }

        val (event, settings) = gate(photographerId, eventId)
        duplicateOf(photographerId, eventId, hash)?.let {
            directOutcome("existing")
            return DirectUploadBeginResponse(mode = "existing", existing = existingPhotoDto(it))
        }
        requireWatermark(settings, event)
        if (!storageService.supportsImmutableDirectUpload ||
            (client == WEBSITE_CLIENT && !storageProperties.directUpload.websiteEnabled)) {
            return DirectUploadBeginResponse(mode = "multipart")
        }

        val intent = try {
            reserveIntent(photographerId, eventId, requestId, hash, type, sizeBytes, originalFilename)
        } catch (error: DataIntegrityViolationException) {
            val existing = requestId?.let {
                photoUploadIntentRepository.findByPhotographerIdAndRequestId(photographerId, it)
            } ?: throw error
            validateReplay(existing, eventId, hash, type, sizeBytes)
            return replayBegin(existing)
        } catch (error: ApiException) {
            if (error.code != ErrorCodes.RATE_LIMITED) throw error
            val existing = requestId?.let {
                photoUploadIntentRepository.findByPhotographerIdAndRequestId(photographerId, it)
            } ?: throw error
            validateReplay(existing, eventId, hash, type, sizeBytes)
            return replayBegin(existing)
        }
        return presign(intent)
    }

    fun commitDirectUpload(
        photographerId: UUID,
        eventId: UUID,
        photoId: UUID,
        key: String,
        contentHash: String,
        sizeBytes: Long? = null,
        contentType: String? = null,
    ): UploadedPhotoDto {
        val timerSample = Timer.start(meterRegistry)
        accessGuard.requireApproved(photographerId, "uploading")
        val hash = contentHash.lowercase()
        val intent = photoUploadIntentRepository.findOwned(photoId, photographerId, eventId)
        if (intent == null) {
            legacyCommittedPhoto(photoId, photographerId, eventId, key, hash)?.let {
                directOutcome("existing")
                return existingPhotoDto(it)
            }
            if (photoRepository.findById(photoId).isPresent) {
                throw ApiException(HttpStatus.NOT_FOUND, ErrorCodes.UPLOAD_NOT_FOUND, "Upload not found.")
            }
            if (key == originalKeyFor(eventId, photoId)) uploadObjectMissing()
            throw ApiException(HttpStatus.NOT_FOUND, ErrorCodes.UPLOAD_NOT_FOUND, "Upload not found.")
        }
        validateCommitRequest(intent, key, hash, sizeBytes, contentType)
        val dto = commitIntent(intent)
        timerSample.stop(meterRegistry.timer("qp.upload.duration"))
        return dto
    }

    fun cleanupDirectUploads() {
        val now = OffsetDateTime.now(clock)
        val batch = PageRequest.of(0, storageProperties.directUpload.cleanupBatchSize)
        try {
            photoUploadIntentRepository.findExpiredCandidateIds(now, batch).forEach { photoId ->
                try {
                    val expired = transactionTemplate.execute {
                        photoUploadIntentRepository.expire(
                            photoId,
                            now.plus(storageProperties.directUpload.cleanupDelay),
                            now,
                        )
                    } ?: 0
                    if (expired == 1) directOutcome("expired")
                } catch (error: Exception) {
                    log.warn("Upload cleanup expire failed intent={} error={}", photoId, error.javaClass.simpleName)
                }
            }

            photoUploadIntentRepository.findCleanupCandidateIds(now, TERMINAL_INTENT_STATES, batch).forEach { photoId ->
                try {
                    cleanupIntent(photoId, now)
                } catch (error: Exception) {
                    log.warn("Upload cleanup settle failed intent={} error={}", photoId, error.javaClass.simpleName)
                }
            }
        } finally {
            refreshPendingMetrics(now)
        }
    }

    private fun refreshPendingMetrics(now: OffsetDateTime) {
        try {
            val metrics = photoUploadIntentRepository.pendingMetrics(now)
            pendingCount.set(metrics?.slots ?: 0)
            pendingBytes.set(metrics?.bytes ?: 0)
            oldestPendingAge.set(
                metrics?.oldestCreatedAt?.let { Duration.between(it, now.toInstant()).seconds.coerceAtLeast(0) } ?: 0,
            )
        } catch (error: Exception) {
            log.warn("Upload pending metrics refresh failed error={}", error.javaClass.simpleName)
        }
    }

    private fun cleanupIntent(photoId: UUID, now: OffsetDateTime) {
        val claimedUntil = now.plus(storageProperties.directUpload.cleanupDelay)
        val claimed = try {
            transactionTemplate.execute {
                photoUploadIntentRepository.claimCleanup(photoId, now, claimedUntil)
            } ?: 0
        } catch (error: Exception) {
            log.warn("Upload cleanup claim failed intent={} error={}", photoId, error.javaClass.simpleName)
            return
        }
        if (claimed != 1) return

        val intent = photoUploadIntentRepository.findById(photoId).orElse(null) ?: return
        var attempted = 0
        var failed = 0
        fun delete(operation: String, key: String) {
            attempted++
            try {
                storageService.delete(key)
            } catch (error: Exception) {
                failed++
                log.warn(
                    "Upload cleanup delete failed operation={} intent={} error={}",
                    operation,
                    photoId,
                    error.javaClass.simpleName,
                )
            }
        }

        delete("staging", intent.stagingKey)
        val protected = try {
            photoRepository.existsByS3Key(intent.originalKey)
        } catch (error: Exception) {
            failed++
            log.warn("Upload cleanup reference check failed intent={} error={}", photoId, error.javaClass.simpleName)
            true
        }
        if (!protected) delete("original", intent.originalKey)

        if (failed > 0) {
            val retried = transactionTemplate.execute {
                photoUploadIntentRepository.retryCleanup(
                    photoId,
                    claimedUntil,
                    now.plus(storageProperties.directUpload.cleanupDelay),
                    now,
                )
            } ?: 0
            if (retried == 1) cleanupOutcome(if (failed < attempted) "partial" else "failed")
            return
        }

        val cleanedAt = intent.cleanedAt ?: now
        val recheckUntil = cleanedAt.plus(storageProperties.directUpload.cleanupRecheckWindow)
        val nextCleanup = if (now >= recheckUntil) {
            null
        } else {
            minOf(now.plus(storageProperties.directUpload.cleanupDelay), recheckUntil)
        }
        val completed = transactionTemplate.execute {
            photoUploadIntentRepository.completeCleanup(
                photoId,
                claimedUntil,
                cleanedAt,
                nextCleanup,
                now,
            )
        } ?: 0
        if (completed == 1) {
            cleanupOutcome(if (protected) "protected" else if (nextCleanup == null) "complete" else "cleaned")
        }
    }

    private fun directOutcome(outcome: String) {
        meterRegistry.counter("qp.upload.direct", "outcome", outcome).increment()
    }

    private fun cleanupOutcome(outcome: String) {
        meterRegistry.counter("qp.upload.cleanup", "outcome", outcome).increment()
    }

    private fun legacyCommittedPhoto(id: UUID, owner: UUID, event: UUID, key: String, hash: String): Photo? =
        photoRepository.findById(id).orElse(null)?.takeIf {
            it.photographerId == owner && it.eventId == event && it.s3Key == key && it.contentHash == hash &&
                it.deletionStartedAt == null
        }

    private fun reserveIntent(
        ownerId: UUID,
        eventId: UUID,
        requestId: UUID?,
        contentHash: String,
        contentType: String,
        sizeBytes: Long,
        originalFilename: String?,
    ): PhotoUploadIntent = checkNotNull(transactionTemplate.execute {
        photographerSettingsRepository.findByUserIdForUpdate(ownerId)
            ?: throw ApiException(HttpStatus.FORBIDDEN, ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED,
                "Submit your photographer verification before uploading.")
        val now = OffsetDateTime.now(clock)
        val quota = photoUploadIntentRepository.activeQuota(ownerId, now)
        val direct = storageProperties.directUpload
        if (quota.slots >= direct.maxPendingPerUser || quota.bytes > direct.maxPendingBytesPerUser - sizeBytes) {
            throw ApiException(HttpStatus.TOO_MANY_REQUESTS, ErrorCodes.RATE_LIMITED,
                "Too many direct uploads are pending. Try again shortly.", retryAfterSeconds = 5)
        }
        val photoId = UUID.randomUUID()
        photoUploadIntentRepository.saveAndFlush(
            PhotoUploadIntent(
                photoId = photoId,
                photographerId = ownerId,
                eventId = eventId,
                requestId = requestId,
                contentHash = contentHash,
                contentType = contentType,
                sizeBytes = sizeBytes,
                originalFilename = sanitizeOriginalFilename(originalFilename),
                stagingKey = stagingKeyFor(ownerId, eventId, photoId),
                originalKey = originalKeyFor(eventId, photoId),
                createdAt = now,
                putExpiresAt = now.plus(direct.putTtl),
                expiresAt = now.plus(direct.intentTtl),
                updatedAt = now,
            ),
        )
    })

    private fun presign(intent: PhotoUploadIntent): DirectUploadBeginResponse {
        val now = OffsetDateTime.now(clock)
        val ttl = Duration.between(now, intent.putExpiresAt)
        if (ttl.isZero || ttl.isNegative) return replayBegin(intent)
        val upload = try {
            storageService.presignedUpload(intent.stagingKey, ttl, intent.contentType, intent.sizeBytes)
        } catch (error: Exception) {
            log.warn("Direct upload presign failed for intent {}: {}", intent.photoId, error.javaClass.simpleName)
            throw ApiException(
                HttpStatus.SERVICE_UNAVAILABLE,
                ErrorCodes.UPLOAD_STORAGE_UNAVAILABLE,
                "Upload storage is temporarily unavailable. Try again.",
                retryAfterSeconds = 5,
            )
        }
        directOutcome("begun")
        return DirectUploadBeginResponse(
            mode = "direct",
            photoId = intent.photoId,
            key = intent.stagingKey,
            uploadUrl = upload.url,
            expiresInSeconds = ttl.seconds.coerceAtLeast(1),
            expiresAt = intent.putExpiresAt,
            intentExpiresAt = intent.expiresAt,
            uploadHeaders = upload.headers,
        )
    }

    private fun validateReplay(
        intent: PhotoUploadIntent,
        eventId: UUID,
        contentHash: String,
        contentType: String,
        sizeBytes: Long,
    ) {
        if (intent.eventId != eventId || intent.contentHash != contentHash ||
            intent.contentType != contentType || intent.sizeBytes != sizeBytes) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.UPLOAD_REQUEST_CONFLICT,
                "That request id was already used for a different upload.")
        }
    }

    private fun replayBegin(intent: PhotoUploadIntent): DirectUploadBeginResponse = when (intent.state) {
        PhotoUploadIntentState.COMMITTED -> DirectUploadBeginResponse(
            mode = "existing",
            existing = committedIntentPhoto(intent),
        )
        PhotoUploadIntentState.REJECTED -> throwStoredRejection(intent.errorCode)
        PhotoUploadIntentState.EXPIRED -> uploadObjectMissing()
        PhotoUploadIntentState.VERIFYING -> {
            val now = OffsetDateTime.now(clock)
            if (intent.leaseUntil?.isAfter(now) == true) uploadInProgress(intent.leaseUntil, now)
            DirectUploadBeginResponse(mode = "existing", existing = commitIntent(intent))
        }
        PhotoUploadIntentState.PENDING -> {
            if (intent.putExpiresAt.isAfter(OffsetDateTime.now(clock))) presign(intent)
            else DirectUploadBeginResponse(mode = "existing", existing = commitIntent(intent))
        }
    }

    private fun validateCommitRequest(
        intent: PhotoUploadIntent,
        key: String,
        contentHash: String,
        sizeBytes: Long?,
        contentType: String?,
    ) {
        if (key != intent.stagingKey) {
            throw ApiException(HttpStatus.UNPROCESSABLE_ENTITY, ErrorCodes.UPLOAD_KEY_MISMATCH,
                "That upload key doesn't match the issued slot.")
        }
        if (contentHash != intent.contentHash ||
            (sizeBytes != null && sizeBytes != intent.sizeBytes) ||
            (contentType != null && !contentType.equals(intent.contentType, ignoreCase = true))) {
            throw ApiException(HttpStatus.UNPROCESSABLE_ENTITY, ErrorCodes.UPLOAD_REQUEST_CONFLICT,
                "The upload metadata doesn't match the issued slot.")
        }
    }

    private fun commitIntent(intent: PhotoUploadIntent): UploadedPhotoDto {
        accessGuard.requireApproved(intent.photographerId, "uploading")
        when (intent.state) {
            PhotoUploadIntentState.COMMITTED -> return committedIntentPhoto(intent)
            PhotoUploadIntentState.REJECTED -> throwStoredRejection(intent.errorCode)
            PhotoUploadIntentState.EXPIRED -> uploadObjectMissing()
            PhotoUploadIntentState.VERIFYING -> {
                val now = OffsetDateTime.now(clock)
                if (intent.leaseUntil?.isAfter(now) == true) uploadInProgress(intent.leaseUntil, now)
            }
            PhotoUploadIntentState.PENDING -> Unit
        }

        val now = OffsetDateTime.now(clock)
        if (!intent.expiresAt.isAfter(now)) {
            val expired = transactionTemplate.execute {
                photoUploadIntentRepository.expire(intent.photoId, cleanupAfter(intent, now), now)
            } ?: 0
            if (expired == 1) directOutcome("expired")
            uploadObjectMissing()
        }
        val (event, settings) = gate(intent.photographerId, intent.eventId)
        requireWatermark(settings, event)
        if (!verificationSlots.tryAcquire()) {
            throw ApiException(HttpStatus.TOO_MANY_REQUESTS, ErrorCodes.RATE_LIMITED,
                "Upload verification is busy. Try again shortly.", retryAfterSeconds = 1)
        }

        val token = UUID.randomUUID()
        try {
            val acquired = transactionTemplate.execute {
                photoUploadIntentRepository.acquireVerification(
                    intent.photoId,
                    token,
                    OffsetDateTime.now(clock).plus(storageProperties.directUpload.verificationLease),
                    OffsetDateTime.now(clock),
                )
            } ?: 0
            if (acquired != 1) return resolveUnacquired(intent)

            try {
                val verifiedHash = validateCandidate(intent)
                val refreshed = gate(intent.photographerId, intent.eventId)
                requireWatermark(refreshed.settings, refreshed.event)
                return persistNew(
                    intent.photoId,
                    intent.eventId,
                    intent.photographerId,
                    intent.originalKey,
                    verifiedHash,
                    refreshed.event,
                    intent,
                    token,
                    intent.originalFilename,
                    intent.sizeBytes,
                )
            } catch (error: ConflictException) {
                if (!rejectIntent(intent, token, error.code)) return resolveUnacquired(intent)
                throw error
            } catch (error: ApiException) {
                if (error.code in TERMINAL_VALIDATION_CODES) {
                    if (!rejectIntent(intent, token, error.code)) return resolveUnacquired(intent)
                } else if (!releaseIntent(
                        intent,
                        token,
                        expireIfPutEnded = error.code == ErrorCodes.UPLOAD_OBJECT_MISSING,
                    )) {
                    return resolveUnacquired(intent)
                }
                throw error
            } catch (_: LeaseLostException) {
                return resolveUnacquired(intent)
            } catch (error: Exception) {
                if (!releaseIntent(intent, token)) return resolveUnacquired(intent)
                log.warn("Direct upload storage operation failed for intent {}: {}", intent.photoId, error.javaClass.simpleName)
                throw ApiException(HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.UPLOAD_STORAGE_UNAVAILABLE,
                    "Upload storage is temporarily unavailable. Try again.", retryAfterSeconds = 5)
            }
        } finally {
            verificationSlots.release()
        }
    }

    private fun validateCandidate(intent: PhotoUploadIntent): String {
        var target = storageService.metadata(intent.originalKey)
        if (target == null) {
            val source = storageService.metadata(intent.stagingKey) ?: uploadObjectMissing()
            validateStoredMetadata(source, intent)
            target = try {
                storageService.copyIfAbsent(
                    intent.stagingKey,
                    intent.originalKey,
                    source.etag.orEmpty(),
                    intent.contentType,
                )
            } catch (error: Exception) {
                storageService.metadata(intent.originalKey) ?: throw error
            }
        }
        validateStoredMetadata(checkNotNull(target), intent)
        val sample = Timer.start(meterRegistry)
        return try {
            photoUploadValidator.validate(
                storageService.open(intent.originalKey),
                intent.contentType,
                intent.sizeBytes,
                intent.contentHash,
            )
        } finally {
            sample.stop(meterRegistry.timer("qp.upload.validation.duration"))
        }
    }

    private fun validateStoredMetadata(
        stored: com.quickpitik.service.storage.StoredObject,
        intent: PhotoUploadIntent,
    ) {
        if (stored.sizeBytes != intent.sizeBytes) {
            throw ApiException(HttpStatus.PAYLOAD_TOO_LARGE, ErrorCodes.PAYLOAD_TOO_LARGE,
                "The stored image size does not match the upload.")
        }
        val missingRequiredType = stored.contentType.isBlank() && storageService.supportsImmutableDirectUpload
        val wrongType = stored.contentType.isNotBlank() &&
            !stored.contentType.equals(intent.contentType, ignoreCase = true)
        if (missingRequiredType || wrongType ||
            !stored.contentEncoding.isNullOrBlank() && !stored.contentEncoding.equals("identity", ignoreCase = true)) {
            throw ApiException(HttpStatus.UNSUPPORTED_MEDIA_TYPE, ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                "The stored image type does not match the upload.")
        }
    }

    private fun resolveUnacquired(intent: PhotoUploadIntent): UploadedPhotoDto {
        val current = photoUploadIntentRepository.findOwned(intent.photoId, intent.photographerId, intent.eventId)
            ?: throw ApiException(HttpStatus.NOT_FOUND, ErrorCodes.UPLOAD_NOT_FOUND, "Upload not found.")
        return when (current.state) {
            PhotoUploadIntentState.COMMITTED -> committedIntentPhoto(current)
            PhotoUploadIntentState.REJECTED -> throwStoredRejection(current.errorCode)
            PhotoUploadIntentState.EXPIRED -> uploadObjectMissing()
            else -> uploadInProgress(current.leaseUntil, OffsetDateTime.now(clock))
        }
    }

    private fun committedIntentPhoto(intent: PhotoUploadIntent): UploadedPhotoDto {
        val resultId = intent.resultPhotoId
            ?: throw NotFoundException(code = ErrorCodes.PHOTO_NOT_FOUND, message = "Photo not found")
        val photo = photoRepository.findById(resultId).orElse(null)
            ?.takeIf {
                it.photographerId == intent.photographerId && it.eventId == intent.eventId &&
                    it.deletionStartedAt == null
            }
            ?: throw NotFoundException(code = ErrorCodes.PHOTO_NOT_FOUND, message = "Photo not found")
        directOutcome("existing")
        return existingPhotoDto(photo)
    }

    private fun rejectIntent(intent: PhotoUploadIntent, token: UUID, errorCode: String): Boolean {
        val rejected = (transactionTemplate.execute {
            photoUploadIntentRepository.reject(
                intent.photoId,
                token,
                errorCode,
                cleanupAfter(intent, OffsetDateTime.now(clock)),
                OffsetDateTime.now(clock),
            )
        } ?: 0) == 1
        if (rejected) directOutcome("rejected")
        return rejected
    }

    private fun releaseIntent(
        intent: PhotoUploadIntent,
        token: UUID,
        expireIfPutEnded: Boolean = false,
    ): Boolean {
        val now = OffsetDateTime.now(clock)
        var expired = false
        val released = (transactionTemplate.execute {
            if (expireIfPutEnded && !intent.putExpiresAt.isAfter(now)) {
                expired = true
                photoUploadIntentRepository.expireMissingAfterPut(
                    intent.photoId,
                    token,
                    cleanupAfter(intent, now),
                    now,
                )
            } else if (intent.expiresAt.isAfter(now)) {
                photoUploadIntentRepository.releaseVerification(intent.photoId, token, now)
            } else {
                expired = true
                photoUploadIntentRepository.expire(intent.photoId, cleanupAfter(intent, now), now)
            }
        } ?: 0) == 1
        if (released && expired) directOutcome("expired")
        return released
    }

    private fun cleanupAfter(intent: PhotoUploadIntent, now: OffsetDateTime): OffsetDateTime =
        maxOf(intent.putExpiresAt, now).plus(storageProperties.directUpload.cleanupDelay)

    private fun throwStoredRejection(code: String?): Nothing = when (code) {
        ErrorCodes.PAYLOAD_TOO_LARGE -> throw ApiException(HttpStatus.PAYLOAD_TOO_LARGE, code, "The stored image size is invalid.")
        ErrorCodes.UPLOAD_CHECKSUM_MISMATCH -> throw ApiException(HttpStatus.UNPROCESSABLE_ENTITY, code, "The uploaded file checksum does not match.")
        ErrorCodes.PHOTO_DUPLICATE_DIFFERENT_EVENT -> throw ConflictException(code, "This photo already exists in another event.")
        else -> throw ApiException(HttpStatus.UNSUPPORTED_MEDIA_TYPE, ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
            "The stored image is not supported.")
    }

    private fun uploadObjectMissing(): Nothing = throw ApiException(
        HttpStatus.UNPROCESSABLE_ENTITY,
        ErrorCodes.UPLOAD_OBJECT_MISSING,
        "The uploaded file wasn't found in storage. Upload it again.",
    )

    private fun uploadInProgress(leaseUntil: OffsetDateTime?, now: OffsetDateTime): Nothing {
        val retry = leaseUntil?.let { Duration.between(now, it).seconds.coerceIn(1, 30) } ?: 1
        throw ApiException(HttpStatus.CONFLICT, ErrorCodes.UPLOAD_IN_PROGRESS,
            "Upload verification is already in progress.", retryAfterSeconds = retry)
    }

    private data class Gate(val event: com.quickpitik.entity.Event, val settings: com.quickpitik.entity.PhotographerSettings)

    /** Everything that must hold before ANY upload path may store a photo. */
    private fun gate(photographerId: UUID, eventId: UUID): Gate {
        val event = eventRepository.findById(eventId).orElse(null)?.takeIf { it.deletedAt == null }
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        if (event.status !in UPLOADABLE_STATUSES) {
            throw ApiException(
                status = HttpStatus.UNPROCESSABLE_ENTITY,
                code = ErrorCodes.EVENT_NOT_UPLOADABLE,
                message = "This event is not accepting uploads.",
            )
        }
        // Photographer-owned events (V46): only the owner uploads into their
        // event. The status gate above still closes a DRAFT (pending review)
        // event; the date window below binds owner + admin events differently.
        val owned = event.createdBy != null
        if (owned && event.createdBy != photographerId) {
            throw ApiException(
                status = HttpStatus.UNPROCESSABLE_ENTITY,
                code = ErrorCodes.EVENT_NOT_UPLOADABLE,
                message = "This event belongs to another photographer.",
            )
        }
        // Upload window (Asia/Manila), enforced here so the UI is never the only
        // gate. Both windows mirror website/src/lib/event-catalog.ts + mobile
        // EventCatalog.kt so the FE upload tile and the backend agree.
        //  - Owner-created event: before/during/after, up to one month after the
        //    event date, then closed. The owner set the date, so no lower bound.
        //  - Admin event: race day + 3 days (4 inclusive); outside that the
        //    gallery is open for sale (or future-dated) and uploads are closed.
        val today = LocalDate.now(clock.withZone(PH_ZONE))
        val closed = if (owned) {
            today.isAfter(event.date.plusMonths(OWNER_UPLOAD_GRACE_MONTHS))
        } else {
            today.isBefore(event.date) || today.isAfter(event.date.plusDays(UPLOAD_GRACE_DAYS - 1L))
        }
        if (closed) {
            throw ApiException(
                status = HttpStatus.UNPROCESSABLE_ENTITY,
                code = ErrorCodes.EVENT_NOT_UPLOADABLE,
                message = if (owned) {
                    "The 1-month upload window for this event has ended."
                } else {
                    "Upload window for this event has closed."
                },
            )
        }
        val settings = accessGuard.requireApproved(photographerId, "uploading")
        return Gate(event, settings)
    }

    /**
     * Per-photographer, cross-event dedup on the SHA-256 of the original bytes.
     * Returns the existing photo for a same-event repeat (idempotent), throws
     * for a different-event repeat, null when the bytes are new.
     */
    private fun duplicateOf(photographerId: UUID, eventId: UUID, contentHash: String): Photo? {
        photoRepository.findFirstByPhotographerIdAndContentHash(photographerId, contentHash)?.let { existing ->
            if (existing.eventId == eventId) {
                meterRegistry.counter("qp.upload.dedup", "outcome", "same_event").increment()
                return existing
            }
            meterRegistry.counter("qp.upload.dedup", "outcome", "different_event").increment()
            val otherEvent = eventRepository.findById(existing.eventId).orElse(null)
            throw ConflictException(
                code = ErrorCodes.PHOTO_DUPLICATE_DIFFERENT_EVENT,
                message = "This photo already exists in your event '${otherEvent?.name ?: "another event"}'.",
            )
        }
        meterRegistry.counter("qp.upload.dedup", "outcome", "new").increment()
        return null
    }

    // A NONE watermark policy (free event, V46) composites nothing, so it is
    // the one case that needs no logo.
    private fun requireWatermark(
        settings: com.quickpitik.entity.PhotographerSettings,
        event: com.quickpitik.entity.Event,
    ) {
        if (event.watermarkPolicy == com.quickpitik.entity.WatermarkPolicy.NONE) return
        if (settings.watermarkS3Key == null) {
            throw ApiException(
                status = HttpStatus.UNPROCESSABLE_ENTITY,
                code = ErrorCodes.WATERMARK_MISSING,
                message = "Your watermark image is missing from storage. Re-upload it in Settings before uploading photos.",
            )
        }
    }

    private fun originalKeyFor(eventId: UUID, photoId: UUID): String =
        "events/$eventId/photos/$photoId/original.jpg"

    private fun stagingKeyFor(photographerId: UUID, eventId: UUID, photoId: UUID): String =
        "staging/photo-uploads/$photographerId/$eventId/$photoId/original"

    /**
     * Row + counters + async triggers for an original that is already durably
     * in storage under [originalKey]. Shared by the multipart and direct paths.
     */
    private fun persistNew(
        photoId: UUID,
        eventId: UUID,
        photographerId: UUID,
        originalKey: String,
        contentHash: String,
        event: com.quickpitik.entity.Event,
        intent: PhotoUploadIntent? = null,
        leaseToken: UUID? = null,
        originalFilename: String? = null,
        originalSizeBytes: Long,
    ): UploadedPhotoDto {
        // PROCESSING until PhotoWatermarkService stores the watermark and flips
        // it LIVE. Runner/public queries filter status = LIVE, so the clean
        // original can never be served to a non-owner while the watermark is
        // pending; the photographer's own library (no status filter) falls back
        // to the original via thumbnailS3Key ?: s3Key.
        val photo = Photo(
            id = photoId,
            eventId = eventId,
            photographerId = photographerId,
            s3Key = originalKey,
            thumbnailS3Key = null,
            watermarkS3Key = null,
            contentHash = contentHash,
            spanWire = PhotoSpan.DEFAULT.wire,
            tone = Random.nextInt(0, 4),
            uploadedAt = OffsetDateTime.now(),
            status = PhotoStatus.PROCESSING,
            pricePhp = event.pricePerPhoto,
        )

        // Faces + bibs are indexed asynchronously off this request by
        // PhotoIndexingService (face enroll + bib OCR via ai-api). The photo is
        // saved as PROCESSING; watermarking publishes it independently. Indexing fills tags and
        // fires a photo.indexed event, with a @Scheduled reconciliation sweep
        // re-driving anything that fails. This replaces the old per-photo
        // blocking ai-api calls that held the request thread + DB transaction
        // for the full inference duration. When ai-api is disabled the photo is
        // marked SKIPPED and never queued.
        photo.indexingStatus =
            if (aiApiProperties.enabled) IndexingStatus.PENDING else IndexingStatus.SKIPPED

        // Presigning is local SigV4 math — done up here so the transaction
        // below makes no storage call at all. The watermark object doesn't
        // exist yet, so this response presigns the ORIGINAL: the upload
        // response goes only to the photo's owner (their own original — the
        // same bytes their download endpoint serves).
        val thumbnailUrl = storageService.presignedGetUrl(originalKey, storageProperties.presignedTtl.thumbnail)

        // Short write transaction: row + counters + event publishes, ~ms. The
        // expensive work (the original storage PUT) already ran above, outside
        // any transaction. Publishing INSIDE the template binds the events to
        // this transaction, so their AFTER_COMMIT listeners
        // (PhotoWatermarkTrigger, PhotoIndexingTrigger) fire at its commit. A
        // persist-phase failure strands the original object already PUT to
        // storage; that orphan window predates this split (PUTs always ran
        // before the flush) and stays accepted.
        try {
            transactionTemplate.execute {
                // Always INSERT. JpaRepository.save merges assigned IDs, which can overwrite an
                // existing original and increment counters twice when a commit response is lost.
                entityManager.persist(photo)
                entityManager.flush()
                entityManager.persist(PhotoMetadata(photoId = photoId,
                    originalFilename = sanitizeOriginalFilename(originalFilename),
                    originalSizeBytes = originalSizeBytes))

                // Atomic counter writes — concurrent uploads during a live marathon are
                // the normal case, not an edge case. The prior read-modify-write pattern
                // lost increments and could PK-collide on first-upload races (H-3 / M-6).
                val now = OffsetDateTime.now()
                eventPhotographerRepository.upsertOnUpload(eventId, photographerId, now)
                eventRepository.incrementPhotoCount(eventId)

                // The photo.published WebSocket frame no longer fires here — it
                // moved to PhotoWatermarkService's flip transaction, so runners are
                // notified only when the watermark URL the frame carries actually
                // resolves (same no-ghost-photos guarantee as before, Q-002, just
                // anchored to the LIVE flip instead of the insert).
                //
                // Generate the watermark + flip PROCESSING → LIVE asynchronously
                // once this transaction commits (AFTER_COMMIT via
                // PhotoWatermarkTrigger). NOT gated on ai-api — always runs.
                eventPublisher.publishEvent(PhotoUploadedForWatermark(photoId = photoId, eventId = eventId))

                // Index faces + bibs asynchronously once this transaction commits
                // (AFTER_COMMIT via PhotoIndexingTrigger), so the request returns
                // without waiting on ai-api inference.
                if (aiApiProperties.enabled) {
                    eventPublisher.publishEvent(PhotoUploadedForIndexing(photoId = photoId, eventId = eventId))
                }

                if (intent != null) {
                    val token = checkNotNull(leaseToken)
                    val settledAt = OffsetDateTime.now(clock)
                    if (photoUploadIntentRepository.complete(
                            intent.photoId,
                            token,
                            photoId,
                            cleanupAfter(intent, settledAt),
                            settledAt,
                        ) != 1) {
                        throw LeaseLostException()
                    }
                }
            }
        } catch (ex: RuntimeException) {
            val violated = generateSequence<Throwable>(ex) { it.cause }
                .filterIsInstance<ConstraintViolationException>().firstOrNull()?.constraintName
            if (violated !in setOf(CONTENT_HASH_CONSTRAINT, "photos_pkey")) throw ex
            // Resolve only AFTER rollback: PostgreSQL rejects reads in an aborted transaction.
            val existing = legacyCommittedPhoto(photoId, photographerId, eventId, originalKey, contentHash)
                ?: duplicateOf(photographerId, eventId, contentHash)
                ?: throw ex
            if (intent == null) {
                if (originalKey != existing.s3Key) runCatching { storageService.delete(originalKey) }
            } else {
                val token = checkNotNull(leaseToken)
                val now = OffsetDateTime.now(clock)
                val completed = transactionTemplate.execute {
                    photoUploadIntentRepository.complete(
                        intent.photoId,
                        token,
                        existing.id,
                        cleanupAfter(intent, now),
                        now,
                    )
                } ?: 0
                if (completed != 1) throw LeaseLostException()
            }
            if (intent != null) directOutcome("existing")
            return existingPhotoDto(existing)
        }

        // Indexing now runs async, so detection isn't known at upload time. The
        // field is eventually-consistent ("pending" until indexing completes);
        // clients treat it as informational. "none" when ai-api is disabled.
        val aiDetectionStatus = if (aiApiProperties.enabled) "pending" else "none"

        if (intent != null) directOutcome("committed")

        return UploadedPhotoDto(
            id = photoId,
            status = "processing",
            uploadedAt = photo.uploadedAt,
            thumbnailUrl = thumbnailUrl,
            span = photo.span.wire,
            aiDetectionStatus = aiDetectionStatus,
        )
    }

    // Pre-flight duplicate check (dedup Phase 2). For each requested content
    // hash, report whether this photographer already has that photo and, if so,
    // whether it's in THIS event (a no-op re-upload the client should skip) or
    // another event (an upload would be rejected — name the holder). Mirrors the
    // upload() boundary exactly: per-photographer, across all events. Read-only,
    // no AI / storage / window / verification gate — it only tells the client
    // what an upload would do, so it can avoid sending the bytes.
    @Transactional(readOnly = true)
    fun checkExisting(photographerId: UUID, eventId: UUID, hashes: List<String>): PhotoExistsResponse {
        // Hashes are matched verbatim against stored hex; normalize case so a
        // client that upper-cases its digest still matches (server stores lower).
        val normalized = hashes.map { it.lowercase() }
        val existingByHash = photoRepository
            .findByPhotographerIdAndContentHashIn(photographerId, normalized)
            .associateBy { it.contentHash }

        // Name the other events in one batch read rather than per-hit.
        val otherEventIds = existingByHash.values
            .map { it.eventId }
            .filterTo(mutableSetOf()) { it != eventId }
        val otherEventNames = if (otherEventIds.isEmpty()) {
            emptyMap()
        } else {
            eventRepository.findAllById(otherEventIds).associate { it.id to it.name }
        }

        val results = normalized.map { hash ->
            val photo = existingByHash[hash]
            when {
                photo == null -> PhotoExistsResult(hash = hash, status = "new")
                photo.eventId == eventId -> PhotoExistsResult(hash = hash, status = "same_event")
                else -> PhotoExistsResult(
                    hash = hash,
                    status = "different_event",
                    eventName = otherEventNames[photo.eventId],
                )
            }
        }
        return PhotoExistsResponse(results = results)
    }

    // Rebuild the upload response from an already-stored photo, for the
    // idempotent same-event duplicate return. Mirrors the live return at the
    // end of upload(): same fields, a fresh presigned thumbnail, and the same
    // eventually-consistent aiDetectionStatus convention.
    private fun existingPhotoDto(photo: Photo): UploadedPhotoDto {
        // A still-PROCESSING duplicate has no watermark object yet — fall back
        // to the original (owner-only response), same as the live return above.
        val thumbnailUrl = storageService.presignedGetUrl(
            photo.thumbnailS3Key ?: photo.s3Key,
            storageProperties.presignedTtl.thumbnail,
        )
        return UploadedPhotoDto(
            id = photo.id,
            status = when (photo.status) {
                PhotoStatus.LIVE -> "live"
                PhotoStatus.HIDDEN -> "hidden"
                PhotoStatus.PROCESSING -> "processing"
            },
            uploadedAt = photo.uploadedAt,
            thumbnailUrl = thumbnailUrl,
            span = photo.span.wire,
            aiDetectionStatus = if (aiApiProperties.enabled) "pending" else "none",
        )
    }

    private fun sanitizeOriginalFilename(raw: String?): String? {
        val basename = raw?.substringAfterLast('/')?.substringAfterLast('\\') ?: return null
        val normalized = Normalizer.normalize(basename, Normalizer.Form.NFC)
            .replace(Regex("[\\p{Cc}\\p{Cf}\\p{Cs}]"), "")
            .trim()
        val codePoints = normalized.codePoints().limit(255).toArray()
        return String(codePoints, 0, codePoints.size).takeIf { it.isNotBlank() }
    }

    private companion object {
        val UPLOADABLE_STATUSES: Set<EventStatus> = setOf(EventStatus.ACTIVE, EventStatus.COMPLETED)
        val ALLOWED_CONTENT_TYPES: Set<String> = setOf("image/jpeg", "image/png", "image/webp")
        val PH_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        const val UPLOAD_GRACE_DAYS = 4
        // Photographer-owned events: the owner may upload up to one calendar
        // month after the event date (before/during/after), then it closes.
        const val OWNER_UPLOAD_GRACE_MONTHS = 1L
        const val WEBSITE_CLIENT = "website"
        val TERMINAL_VALIDATION_CODES = setOf(
            ErrorCodes.PAYLOAD_TOO_LARGE,
            ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
            ErrorCodes.UPLOAD_CHECKSUM_MISMATCH,
        )
        val TERMINAL_INTENT_STATES = listOf(
            PhotoUploadIntentState.COMMITTED,
            PhotoUploadIntentState.REJECTED,
            PhotoUploadIntentState.EXPIRED,
        )

        // The V24 partial unique index. The race backstop only treats THIS
        // constraint as a duplicate; any other violation is rethrown.
        const val CONTENT_HASH_CONSTRAINT = "uq_photos_photographer_content_hash"
    }

    private class LeaseLostException : RuntimeException()
}
