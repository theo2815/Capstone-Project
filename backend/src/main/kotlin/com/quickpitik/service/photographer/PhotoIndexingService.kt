package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoBibEmbed
import com.quickpitik.entity.PhotoFacePersonEmbed
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.storage.StorageService
import com.quickpitik.websocket.PhotoIndexedEvent
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// Async ai-api indexing of an uploaded photo: one face enroll + one bib OCR.
// Called cross-bean from PhotoIndexingTrigger (so the @Transactional proxy
// applies — a self-invocation would silently run without a transaction).
@Service
class PhotoIndexingService(
    private val photoRepository: PhotoRepository,
    private val storageService: StorageService,
    private val aiApiClient: AiApiClient,
    private val aiApiProperties: AiApiProperties,
    private val eventPublisher: ApplicationEventPublisher,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional
    fun index(photoId: UUID) {
        val photo = photoRepository.findById(photoId).orElse(null) ?: return
        if (photo.indexingStatus == IndexingStatus.INDEXED ||
            photo.indexingStatus == IndexingStatus.SKIPPED
        ) {
            return
        }
        if (!aiApiProperties.enabled) {
            photo.indexingStatus = IndexingStatus.SKIPPED
            photoRepository.save(photo)
            return
        }

        photo.indexingAttempts += 1
        // Retry-safety: a re-run is a clean slate (no duplicate embeddings, no
        // orphaned ai-api persons). No-op on the first attempt.
        clearPriorResults(photo)

        val bytes = try {
            storageService.getBytes(photo.s3Key)
        } catch (ex: Exception) {
            log.warn("Indexing {}: original bytes unavailable ({}); marking FAILED", photoId, ex.message)
            photo.indexingStatus = IndexingStatus.FAILED
            photo.indexingError = "original image unavailable: ${ex.message}"
            photoRepository.save(photo)
            return
        }
        // Index the ORIGINAL, never the watermarked image — the diagonal
        // anti-piracy overlay would corrupt face + bib detection.
        val contentType = sniffContentType(bytes)
        val filename = "$photoId.jpg"

        photo.indexingError = null
        val facesOk = enrollFaces(photo, bytes, contentType, filename)
        val bibsOk = recognizeBibs(photo, bytes, contentType, filename)

        photo.indexingStatus = when {
            facesOk && bibsOk -> IndexingStatus.INDEXED
            facesOk || bibsOk -> IndexingStatus.PARTIAL
            else -> IndexingStatus.FAILED
        }
        photo.indexedAt =
            if (photo.indexingStatus == IndexingStatus.INDEXED) OffsetDateTime.now() else null
        photoRepository.save(photo)

        // Notify the live gallery even on PARTIAL so the bib (if any) surfaces.
        if (facesOk || bibsOk) {
            eventPublisher.publishEvent(
                PhotoIndexedEvent(
                    eventId = photo.eventId,
                    payload = mapOf(
                        "type" to "photo.indexed",
                        "photo" to mapOf(
                            "id" to photo.id.toString(),
                            "bib" to photo.bibs.minByOrNull { it.bibNumber }?.bibNumber,
                            "indexingStatus" to photo.indexingStatus.name,
                        ),
                    ),
                ),
            )
        }
    }

    // One enroll per photo: the photo IS the ai "person", and every face in it
    // (each a distinct runner) is stored under that single person_id, so any of
    // them matches at search time. NO_FACES / LOW_QUALITY are benign — the call
    // ran, there was simply nothing usable to enroll.
    private fun enrollFaces(photo: Photo, bytes: ByteArray, contentType: String, filename: String): Boolean =
        try {
            val enroll = aiApiClient.facesEnroll(
                file = bytes,
                contentType = contentType,
                filename = filename,
                personName = photo.id.toString(),
                personId = null,
                eventId = photo.eventId,
            )
            photo.facePersons.add(PhotoFacePersonEmbed(faceIndex = 0, aiPersonId = enroll.person_id))
            true
        } catch (ex: AiApiException) {
            if (ex.aiCode == "NO_FACES" || ex.aiCode == "LOW_QUALITY") {
                true
            } else {
                log.warn("Face enroll failed for photo {}: {}", photo.id, ex.message)
                photo.indexingError = "faces: ${ex.message}"
                false
            }
        } catch (ex: Exception) {
            log.warn("Face enroll failed for photo {}: {}", photo.id, ex.message)
            photo.indexingError = "faces: ${ex.message}"
            false
        }

    private fun recognizeBibs(photo: Photo, bytes: ByteArray, contentType: String, filename: String): Boolean =
        try {
            val result = aiApiClient.bibsRecognize(bytes, contentType, filename)
            // Group only the QUALIFYING detections by normalized bib number and
            // store the max confidence among them — a re-lookup over the full
            // (unfiltered) list could otherwise persist a below-threshold value
            // when the same bib appears more than once in one photo.
            result.detections
                .filter { it.confidence >= aiApiProperties.bibConfidenceThresholdDefault }
                .groupBy { it.bib_number.trim().uppercase() }
                .filterKeys { it.isNotEmpty() }
                .forEach { (bibNumber, detections) ->
                    photo.bibs.add(
                        PhotoBibEmbed(
                            bibNumber = bibNumber,
                            ocrConfidence = BigDecimal.valueOf(detections.maxOf { it.confidence }),
                        ),
                    )
                }
            true
        } catch (ex: Exception) {
            log.warn("Bib recognize failed for photo {}: {}", photo.id, ex.message)
            photo.indexingError = listOfNotNull(photo.indexingError, "bibs: ${ex.message}").joinToString("; ")
            false
        }

    // Drops any ai person + tags from a prior attempt. deleteFacesPerson is the
    // same primitive used for GDPR erasure when a photo is removed.
    private fun clearPriorResults(photo: Photo) {
        photo.facePersons.forEach { fp ->
            runCatching { aiApiClient.deleteFacesPerson(fp.aiPersonId) }
                .onFailure { log.warn("ai person cleanup failed for {}: {}", fp.aiPersonId, it.message) }
        }
        photo.facePersons.clear()
        photo.bibs.clear()
    }

    // ai-api validates by magic bytes, not Content-Type, but the multipart part
    // still needs a sensible media type. Sniff it; default to JPEG.
    private fun sniffContentType(bytes: ByteArray): String = when {
        bytes.size >= 3 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xD8.toByte() && bytes[2] == 0xFF.toByte() -> "image/jpeg"
        bytes.size >= 8 && bytes[0] == 0x89.toByte() && bytes[1] == 0x50.toByte() -> "image/png"
        bytes.size >= 12 && bytes[0] == 'R'.code.toByte() && bytes[3] == 'F'.code.toByte() -> "image/webp"
        else -> "image/jpeg"
    }
}
