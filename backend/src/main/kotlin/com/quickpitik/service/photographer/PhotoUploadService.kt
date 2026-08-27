package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.PhotoExistsResponse
import com.quickpitik.dto.photographer.PhotoExistsResult
import com.quickpitik.dto.photographer.UploadedPhotoDto
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoSpan
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.image.ImagePixelGuard
import com.quickpitik.service.storage.StorageService
import com.quickpitik.websocket.PhotoPublishedEvent
import io.micrometer.core.instrument.MeterRegistry
import io.micrometer.core.instrument.Timer
import org.hibernate.exception.ConstraintViolationException
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
import org.springframework.web.multipart.MultipartFile
import java.security.MessageDigest
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.HexFormat
import java.util.UUID
import kotlin.random.Random

@Service
class PhotoUploadService(
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val photoRepository: PhotoRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val eventRepository: EventRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
    private val aiApiProperties: AiApiProperties,
    private val watermarkService: WatermarkService,
    private val watermarkLogoCache: WatermarkLogoCache,
    private val eventPublisher: ApplicationEventPublisher,
    private val transactionTemplate: TransactionTemplate,
    private val meterRegistry: MeterRegistry,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // Deliberately NOT @Transactional: the watermark fetch, JPEG decode +
    // composite, and the two object-storage PUTs below take seconds each and
    // would pin a Hikari connection (pool of 10) for the whole ride — marathon
    // -day concurrent uploads exhausted the pool that way. Only the short
    // persist block at the end runs in a transaction (TransactionTemplate).
    fun upload(photographerId: UUID, eventId: UUID, file: MultipartFile): UploadedPhotoDto {
        // Times SUCCESSFUL new-photo uploads only (dedup short-circuits and
        // validation rejects return before the stop) — the number that matters
        // for marathon-day capacity planning.
        val timerSample = Timer.start(meterRegistry)
        val event = eventRepository.findById(eventId).orElse(null)?.takeIf { it.deletedAt == null }
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        if (event.status !in UPLOADABLE_STATUSES) {
            throw ApiException(
                status = HttpStatus.UNPROCESSABLE_ENTITY,
                code = ErrorCodes.EVENT_NOT_UPLOADABLE,
                message = "This event is not accepting uploads.",
            )
        }
        // Photographer upload window is race day + 3 days (4 days inclusive,
        // Asia/Manila). Outside that window the gallery is open for sale
        // (or future-dated and not yet opened to runners) and new uploads
        // are closed — mirrors website/src/lib/event-catalog.ts
        // canUploadToEvent so the FE upload tile and the backend agree.
        val today = LocalDate.now(PH_ZONE)
        if (today.isBefore(event.date) || today.isAfter(event.date.plusDays(UPLOAD_GRACE_DAYS - 1L))) {
            throw ApiException(
                status = HttpStatus.UNPROCESSABLE_ENTITY,
                code = ErrorCodes.EVENT_NOT_UPLOADABLE,
                message = "Upload window for this event has closed.",
            )
        }
        val photographer = userRepository.findById(photographerId).orElse(null)
            ?: throw NotFoundException(code = ErrorCodes.USER_NOT_FOUND, message = "User not found")
        if (photographer.suspendedAt != null) {
            throw ApiException(
                status = HttpStatus.FORBIDDEN,
                code = ErrorCodes.ACCOUNT_SUSPENDED,
                message = "Your account is suspended. Contact support to appeal before uploading.",
            )
        }
        val settings = photographerSettingsRepository.findById(photographerId).orElse(null)
            ?: throw ApiException(
                status = HttpStatus.FORBIDDEN,
                code = ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED,
                message = "Submit your photographer verification before uploading.",
            )
        if (settings.verificationStatus != VerificationStatus.APPROVED) {
            throw ApiException(
                status = HttpStatus.FORBIDDEN,
                code = ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED,
                message = "Submit your photographer verification before uploading.",
            )
        }

        val contentType = file.contentType?.lowercase()
        if (file.isEmpty || contentType == null || contentType !in ALLOWED_CONTENT_TYPES) {
            throw ApiException(
                status = HttpStatus.UNSUPPORTED_MEDIA_TYPE,
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "Upload must be image/jpeg, image/png, or image/webp.",
            )
        }

        val bytes = file.bytes

        // Decompression-bomb guard: header-only dimension check BEFORE the full
        // ImageIO decode below. The 25 MB multipart cap bounds the compressed
        // bytes, not the decoded raster — a bomb under 25 MB is unbounded heap.
        if (ImagePixelGuard.exceedsPixelBudget(bytes)) {
            throw ApiException(
                status = HttpStatus.UNSUPPORTED_MEDIA_TYPE,
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "Image dimensions exceed the supported maximum.",
            )
        }

        // Duplicate detection (enterprise dedup). A photo's identity is the
        // SHA-256 of its ORIGINAL bytes, hashed HERE — before watermarking,
        // which would change them — so a client- and server-side hash of the
        // same file agree. Boundary is per-photographer across all events: the
        // same shot can't be uploaded twice. A same-event re-upload (the
        // stop-on-mobile / continue-on-web case) returns the existing photo
        // idempotently; a different-event re-upload is rejected. The partial
        // unique index (photographer_id, content_hash) from V24 is the
        // race-safe backstop (see the saveAndFlush below).
        val contentHash = sha256Hex(bytes)
        photoRepository.findFirstByPhotographerIdAndContentHash(photographerId, contentHash)?.let { existing ->
            if (existing.eventId == eventId) {
                meterRegistry.counter("qp.upload.dedup", "outcome", "same_event").increment()
                return existingPhotoDto(existing)
            }
            meterRegistry.counter("qp.upload.dedup", "outcome", "different_event").increment()
            val otherEvent = eventRepository.findById(existing.eventId).orElse(null)
            throw ConflictException(
                code = ErrorCodes.PHOTO_DUPLICATE_DIFFERENT_EVENT,
                message = "This photo already exists in your event '${otherEvent?.name ?: "another event"}'.",
            )
        }
        meterRegistry.counter("qp.upload.dedup", "outcome", "new").increment()

        // Blur culling is desktop-only (BatchMyPhotos). Web upload assumes
        // photos have already been culled by the photographer's desktop
        // workflow — see backend/decisions 2026-05-18 "Blur removed from
        // BE upload" + website/decisions 2026-05-06 "Blur removed from web."

        val photoId = UUID.randomUUID()
        val originalKey = "events/$eventId/photos/$photoId/original.jpg"
        val watermarkKey = "events/$eventId/photos/$photoId/watermark.jpg"

        // Photographer's watermark IMAGE (logo) gets composited onto every
        // upload — bottom-right corner (identification) + diagonal center
        // (anti-piracy). Verification gate at line 92 + PhotographerSettings
        // collectMissing line 372-374 guarantee watermarkS3Key is non-null at
        // this point; !! is safe. WatermarkService handles transparent PNG +
        // opaque JPEG watermarks identically (alpha-aware composite).
        //
        // The defensive try/catch covers a real-world drift case observed
        // 2026-05-18: DB pointer + disk file went out of sync (DB referenced
        // a key whose bytes no longer existed on disk — partial nuke, manual
        // file delete, or a race in uploadWatermark's put → save → delete
        // sequence). Surface a clean 422 so the FE prompts the photographer
        // to re-upload via /dashboard/settings rather than dumping a 500.
        val watermarkBytes = try {
            watermarkLogoCache.get(settings.watermarkS3Key!!)
        } catch (ex: java.io.IOException) {
            // Broadened from NoSuchFileException, which only covered the local-fs
            // backend. Under S3 a missing object, timeout, or permission denial
            // surfaces as a different IOException and used to escape as a 500 —
            // the photographer got "internal error" instead of the actionable
            // "re-upload your watermark" 422 this branch exists to give them.
            log.warn(
                "Watermark unreadable for photographer {} (key {}): {}",
                photographerId,
                settings.watermarkS3Key,
                ex.message,
            )
            throw ApiException(
                status = HttpStatus.UNPROCESSABLE_ENTITY,
                code = ErrorCodes.WATERMARK_MISSING,
                message = "Your watermark image is missing from storage. Re-upload it in Settings before uploading photos.",
            )
        }

        val watermarked = try {
            watermarkService.processThumbnail(bytes, watermarkBytes)
        } catch (ex: IllegalArgumentException) {
            // Magic-byte mismatch — Content-Type lied, or the watermark image
            // can't be decoded (stale storage). Reject with the same code as
            // content-type filtering so the FE handles them as one.
            throw ApiException(
                status = HttpStatus.UNSUPPORTED_MEDIA_TYPE,
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "Could not decode image bytes.",
            )
        } catch (ex: java.io.IOException) {
            // ImageIO.read signals a TRUNCATED or structurally corrupt stream
            // with IOException (usually IIOException) rather than returning
            // null, and only the null path raised IllegalArgumentException
            // above — so a partial JPEG escaped as a 500.
            //
            // This is not hypothetical: the camera-card import path pulls bytes
            // off a DSLR over PTP, and a pull interrupted by a cable knock or a
            // detach mid-frame produces exactly these bytes. The client should
            // be told the file is unusable so it can stop retrying it, which a
            // 500 does not do.
            log.warn("Undecodable image bytes on upload for photographer {}: {}", photographerId, ex.message)
            throw ApiException(
                status = HttpStatus.UNSUPPORTED_MEDIA_TYPE,
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "Could not decode image bytes.",
            )
        }

        storageService.put(originalKey, bytes, contentType)
        storageService.put(watermarkKey, watermarked, "image/jpeg")

        val photo = Photo(
            id = photoId,
            eventId = eventId,
            photographerId = photographerId,
            s3Key = originalKey,
            thumbnailS3Key = watermarkKey,
            watermarkS3Key = watermarkKey,
            contentHash = contentHash,
            spanWire = PhotoSpan.DEFAULT.wire,
            tone = Random.nextInt(0, 4),
            uploadedAt = OffsetDateTime.now(),
            status = PhotoStatus.LIVE,
            pricePhp = event.pricePerPhoto,
        )

        // Faces + bibs are indexed asynchronously off this request by
        // PhotoIndexingService (face enroll + bib OCR via ai-api). The photo is
        // saved + LIVE immediately; indexing fills in tags within seconds and
        // fires a photo.indexed event, with a @Scheduled reconciliation sweep
        // re-driving anything that fails. This replaces the old per-photo
        // blocking ai-api calls that held the request thread + DB transaction
        // for the full inference duration. When ai-api is disabled the photo is
        // marked SKIPPED and never queued.
        photo.indexingStatus =
            if (aiApiProperties.enabled) IndexingStatus.PENDING else IndexingStatus.SKIPPED

        // Presigning is local SigV4 math — done up here so the transaction
        // below makes no storage call at all.
        val thumbnailUrl = storageService.presignedGetUrl(watermarkKey, storageProperties.presignedTtl.thumbnail)

        // Short write transaction: row + counters + event publishes, ~ms. The
        // expensive work (watermark fetch, composite, both storage PUTs) already
        // ran above, outside any transaction. Publishing INSIDE the template
        // binds both events to this transaction, so their AFTER_COMMIT listeners
        // (PhotoPublishedBroadcaster, PhotoIndexingTrigger) fire at its commit —
        // exactly as under the old method-level @Transactional. A persist-phase
        // failure strands the two objects already PUT to storage; that orphan
        // window predates this split (PUTs always ran before the flush) and
        // stays accepted.
        transactionTemplate.execute {
            // saveAndFlush (not save) so a concurrent identical-bytes upload that
            // slipped past the pre-check above trips the (photographer_id,
            // content_hash) unique index HERE rather than at commit — the
            // authoritative, race-safe dedup backstop. A duplicate is the only
            // unique constraint a valid photo insert can violate (the id is a fresh
            // random UUID), so translate it to a terminal duplicate conflict.
            try {
                photoRepository.saveAndFlush(photo)
            } catch (ex: DataIntegrityViolationException) {
                // Defense-in-depth: a fresh-UUID id and fully-populated columns mean
                // the dedup index is the only unique constraint a valid insert can
                // violate today, but a FUTURE constraint must not be silently
                // mislabeled a duplicate. Walk the cause chain to the Hibernate
                // ConstraintViolationException (Spring may wrap it a layer deep); if
                // it positively names a DIFFERENT constraint, rethrow it as the
                // genuine integrity fault it is. A null/unknown name keeps the old
                // behavior (treat as the dedup race).
                val violated = generateSequence(ex.cause) { it.cause }
                    .filterIsInstance<ConstraintViolationException>()
                    .firstOrNull()
                    ?.constraintName
                if (violated != null && !violated.equals(CONTENT_HASH_CONSTRAINT, ignoreCase = true)) {
                    throw ex
                }
                // Same- vs different-event can't be re-resolved here: the unique
                // violation has aborted this transaction (Postgres 25P02), so any
                // follow-up read would fail. Emit the same-event conflict; if it was
                // actually a different-event race the photographer's retry hits the
                // pre-check above — which now sees the committed row — and gets the
                // precise "already in event X" message. The different-event race
                // (same bytes, two events, same instant) is pathological and
                // self-heals on that retry.
                throw ConflictException(
                    code = ErrorCodes.PHOTO_DUPLICATE_SAME_EVENT,
                    message = "This photo was just uploaded. Refresh to see it.",
                )
            }

            // Atomic counter writes — concurrent uploads during a live marathon are
            // the normal case, not an edge case. The prior read-modify-write pattern
            // lost increments and could PK-collide on first-upload races (H-3 / M-6).
            val now = OffsetDateTime.now()
            eventPhotographerRepository.upsertOnUpload(eventId, photographerId, now)
            eventRepository.incrementPhotoCount(eventId)

            // Publish via Spring event so the broadcast fires AFTER_COMMIT
            // (PhotoPublishedBroadcaster). Inline broadcast risks ghost photos:
            // runners receive photo.published, a rollback discards the row, the
            // FE 404s on the next fetch. Q-002.
            eventPublisher.publishEvent(
                PhotoPublishedEvent(
                    eventId = eventId,
                    payload = mapOf(
                        "type" to "photo.published",
                        "photo" to mapOf(
                            "id" to photoId.toString(),
                            "bib" to (photo.bibs.minByOrNull { it.bibNumber }?.bibNumber),
                            "tone" to photo.tone,
                            "span" to photo.span.wire,
                            "imageUrl" to thumbnailUrl,
                            "uploadedAt" to photo.uploadedAt.toString(),
                        ),
                    ),
                ),
            )

            // Index faces + bibs asynchronously once this transaction commits
            // (AFTER_COMMIT via PhotoIndexingTrigger), so the request returns
            // without waiting on ai-api inference.
            if (aiApiProperties.enabled) {
                eventPublisher.publishEvent(PhotoUploadedForIndexing(photoId = photoId, eventId = eventId))
            }
        }

        // Indexing now runs async, so detection isn't known at upload time. The
        // field is eventually-consistent ("pending" until indexing completes);
        // clients treat it as informational. "none" when ai-api is disabled.
        val aiDetectionStatus = if (aiApiProperties.enabled) "pending" else "none"

        timerSample.stop(meterRegistry.timer("qp.upload.duration"))
        return UploadedPhotoDto(
            id = photoId,
            status = "live",
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

    private fun sha256Hex(bytes: ByteArray): String =
        HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))

    // Rebuild the upload response from an already-stored photo, for the
    // idempotent same-event duplicate return. Mirrors the live return at the
    // end of upload(): same fields, a fresh presigned thumbnail, and the same
    // eventually-consistent aiDetectionStatus convention.
    private fun existingPhotoDto(photo: Photo): UploadedPhotoDto {
        val thumbnailUrl = storageService.presignedGetUrl(
            photo.thumbnailS3Key!!,
            storageProperties.presignedTtl.thumbnail,
        )
        return UploadedPhotoDto(
            id = photo.id,
            status = when (photo.status) {
                PhotoStatus.LIVE -> "live"
                PhotoStatus.HIDDEN -> "hidden"
                PhotoStatus.PROCESSING -> "live"
            },
            uploadedAt = photo.uploadedAt,
            thumbnailUrl = thumbnailUrl,
            span = photo.span.wire,
            aiDetectionStatus = if (aiApiProperties.enabled) "pending" else "none",
        )
    }

    private companion object {
        val UPLOADABLE_STATUSES: Set<EventStatus> = setOf(EventStatus.ACTIVE, EventStatus.COMPLETED)
        val ALLOWED_CONTENT_TYPES: Set<String> = setOf("image/jpeg", "image/png", "image/webp")
        val PH_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        const val UPLOAD_GRACE_DAYS = 4

        // The V24 partial unique index. The race backstop only treats THIS
        // constraint as a duplicate; any other violation is rethrown.
        const val CONTENT_HASH_CONSTRAINT = "uq_photos_photographer_content_hash"
    }
}
