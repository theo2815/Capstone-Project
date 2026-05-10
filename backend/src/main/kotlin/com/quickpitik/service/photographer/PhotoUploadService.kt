package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.UploadedPhotoDto
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
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
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.storage.StorageService
import com.quickpitik.websocket.PhotoPublishedEvent
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.web.multipart.MultipartFile
import java.math.BigDecimal
import java.time.OffsetDateTime
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

        // Blur gate runs before any storage I/O so blurry uploads cost the
        // photographer round-trip latency only — no storage churn, no DB row.
        val blurResult = try {
            aiApiClient.blurDetect(bytes, contentType, filename)
        } catch (ex: AiApiException) {
            // Bubble up via dedicated handler → 503 AI_API_UNAVAILABLE envelope.
            throw ex
        }
        if (blurResult.is_blurry) {
            throw ApiException(
                status = HttpStatus.UNPROCESSABLE_ENTITY,
                code = ErrorCodes.BLUR_REJECTED,
                message = "Photo did not pass blur quality check.",
            )
        }

        val photoId = UUID.randomUUID()
        val originalKey = "events/$eventId/photos/$photoId/original.jpg"
        val watermarkKey = "events/$eventId/photos/$photoId/watermark.jpg"

        val watermarkLabel = settings.watermarkLabel?.takeIf { it.isNotBlank() }
            ?: settings.brandName?.takeIf { it.isNotBlank() }
            ?: "QUICKPITIK"

        val watermarked = try {
            watermarkService.processThumbnail(bytes, watermarkLabel)
        } catch (ex: IllegalArgumentException) {
            // Magic-byte mismatch — Content-Type lied. Reject with the same
            // code as content-type filtering so the FE handles them as one.
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
            blurScore = blurResult.metrics.laplacian_variance.toBigDecimal(),
            spanWire = PhotoSpan.DEFAULT.wire,
            tone = Random.nextInt(0, 4),
            uploadedAt = OffsetDateTime.now(),
            status = PhotoStatus.LIVE,
            pricePhp = event.pricePerPhoto,
        )

        // Faces — best-effort. A failure here doesn't block the upload; the
        // photo is still searchable by bib number, just not by selfie.
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
            log.warn("Faces detect failed for upload {}: {}", photoId, ex.message)
        }

        // Bibs — best-effort, filtered by configurable confidence floor.
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
            log.warn("Bibs recognize failed for upload {}: {}", photoId, ex.message)
        }

        photoRepository.save(photo)

        val ep = eventPhotographerRepository
            .findById(EventPhotographerId(eventId = eventId, photographerId = photographerId))
            .orElseGet {
                EventPhotographer(
                    id = EventPhotographerId(eventId = eventId, photographerId = photographerId),
                )
            }
        val now = OffsetDateTime.now()
        ep.photoCount += 1
        ep.lastUploadAt = now
        if (ep.firstUploadAt == null) ep.firstUploadAt = now
        eventPhotographerRepository.save(ep)

        // Bump the event-wide counter so /events/[slug] hero stays accurate.
        event.photoCount += 1
        eventRepository.save(event)

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

        return UploadedPhotoDto(
            id = photoId,
            status = "live",
            uploadedAt = photo.uploadedAt,
            thumbnailUrl = thumbnailUrl,
            span = photo.span.wire,
        )
    }

    private companion object {
        val UPLOADABLE_STATUSES: Set<EventStatus> = setOf(EventStatus.ACTIVE, EventStatus.COMPLETED)
        val ALLOWED_CONTENT_TYPES: Set<String> = setOf("image/jpeg", "image/png", "image/webp")
    }
}
