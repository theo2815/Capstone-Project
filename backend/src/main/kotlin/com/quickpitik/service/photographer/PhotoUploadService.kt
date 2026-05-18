package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.UploadedPhotoDto
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoBibEmbed
import com.quickpitik.entity.PhotoFacePersonEmbed
import com.quickpitik.entity.PhotoSpan
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.storage.StorageService
import com.quickpitik.websocket.PhotoPublishedEvent
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.web.multipart.MultipartFile
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
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
    private val aiApiClient: AiApiClient,
    private val aiApiProperties: AiApiProperties,
    private val watermarkService: WatermarkService,
    private val eventPublisher: ApplicationEventPublisher,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional
    fun upload(photographerId: UUID, eventId: UUID, file: MultipartFile): UploadedPhotoDto {
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
        val filename = file.originalFilename ?: "upload.jpg"

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
            storageService.getBytes(settings.watermarkS3Key!!)
        } catch (ex: java.nio.file.NoSuchFileException) {
            log.warn(
                "Watermark file missing for photographer {} (key {})",
                photographerId,
                settings.watermarkS3Key,
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
            spanWire = PhotoSpan.DEFAULT.wire,
            tone = Random.nextInt(0, 4),
            uploadedAt = OffsetDateTime.now(),
            status = PhotoStatus.LIVE,
            pricePhp = event.pricePerPhoto,
        )

        // Faces + bibs — best-effort. Both are gated by `app.ai-api.enabled`
        // so feature development can run without ai-api on. When disabled,
        // facesOk + bibsOk stay false → aiDetectionStatus = "none" in the
        // response, photo still uploads, just without face embeddings or bib
        // OCR. When ai-api comes online (AI_API_ENABLED=true), the existing
        // try/catch graceful degradation kicks in for transient failures.
        // Inner enroll failures are logged via runCatching's onFailure.
        var facesOk = false
        var bibsOk = false
        if (aiApiProperties.enabled) {
            facesOk = true
            try {
                val facesResult = aiApiClient.facesDetect(bytes, contentType, filename)
                facesResult.faces.forEachIndexed { index, _ ->
                    val personId = "$photoId:$index"
                    runCatching {
                        aiApiClient.facesEnroll(
                            file = bytes,
                            contentType = contentType,
                            filename = filename,
                            personName = photoId.toString(),
                            personId = personId,
                            eventId = eventId,
                        )
                        photo.facePersons.add(PhotoFacePersonEmbed(faceIndex = index, aiPersonId = personId))
                    }.onFailure { log.warn("Face enroll failed for {}: {}", personId, it.message) }
                }
            } catch (ex: Exception) {
                facesOk = false
                log.warn("Faces detect failed for upload {}: {}", photoId, ex.message)
            }

            bibsOk = true
            try {
                val bibsResult = aiApiClient.bibsRecognize(bytes, contentType, filename)
                bibsResult.detections
                    .filter { it.confidence >= aiApiProperties.bibConfidenceThresholdDefault }
                    .map { it.bib_number.trim().uppercase() }
                    .filter { it.isNotEmpty() }
                    .distinct()
                    .forEach { bibNumber ->
                        photo.bibs.add(
                            PhotoBibEmbed(
                                bibNumber = bibNumber,
                                ocrConfidence = BigDecimal.valueOf(
                                    bibsResult.detections.first { it.bib_number.trim().uppercase() == bibNumber }.confidence,
                                ),
                            ),
                        )
                    }
            } catch (ex: Exception) {
                bibsOk = false
                log.warn("Bibs recognize failed for upload {}: {}", photoId, ex.message)
            }
        } else {
            log.debug("ai-api disabled; skipping faces + bibs for photo {}", photoId)
        }

        photoRepository.save(photo)

        // Atomic counter writes — concurrent uploads during a live marathon are
        // the normal case, not an edge case. The prior read-modify-write pattern
        // lost increments and could PK-collide on first-upload races (H-3 / M-6).
        val now = OffsetDateTime.now()
        eventPhotographerRepository.upsertOnUpload(eventId, photographerId, now)
        eventRepository.incrementPhotoCount(eventId)

        val thumbnailUrl = storageService.presignedGetUrl(watermarkKey, storageProperties.presignedTtl.thumbnail)

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

        val aiDetectionStatus = when {
            facesOk && bibsOk -> "ok"
            !facesOk && bibsOk -> "faces_unavailable"
            facesOk && !bibsOk -> "bibs_unavailable"
            else -> "none"
        }

        return UploadedPhotoDto(
            id = photoId,
            status = "live",
            uploadedAt = photo.uploadedAt,
            thumbnailUrl = thumbnailUrl,
            span = photo.span.wire,
            aiDetectionStatus = aiDetectionStatus,
        )
    }

    private companion object {
        val UPLOADABLE_STATUSES: Set<EventStatus> = setOf(EventStatus.ACTIVE, EventStatus.COMPLETED)
        val ALLOWED_CONTENT_TYPES: Set<String> = setOf("image/jpeg", "image/png", "image/webp")
        val PH_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        const val UPLOAD_GRACE_DAYS = 4
    }
}
