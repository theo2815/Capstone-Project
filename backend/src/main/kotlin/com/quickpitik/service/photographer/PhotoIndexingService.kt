package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.AiProperties
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoBibEmbed
import com.quickpitik.entity.PhotoFacePersonEmbed
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.service.storage.StorageService
import com.quickpitik.websocket.PhotoIndexedEvent
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// Async AI indexing of an uploaded photo: one face enroll + one bib OCR.
// Runs in three phases so no DB connection is held across storage/AI network
// I/O — the old single @Transactional pinned one for the full inference
// duration (~37s+ per photo against a down provider, times the 8-thread
// imageProcessing pool plus the reconcile sweep):
//   A. gate + snapshot (plain reads),
//   B. I/O — prior-person cleanup, S3 GET, enroll + bib calls, no tx,
//   C. short write tx (TransactionTemplate) — state machine + event publish,
//      so the AFTER_COMMIT broadcaster fires on the template's commit.
@Service
class PhotoIndexingService(
    private val photoRepository: PhotoRepository,
    private val storageService: StorageService,
    private val aiApiClient: FaceBibProvider,
    private val aiApiProperties: AiApiProperties,
    private val aiProperties: AiProperties,
    private val eventPublisher: ApplicationEventPublisher,
    private val transactionTemplate: TransactionTemplate,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun index(photoId: UUID) {
        // Phase A — gate on a plain read. The detached entity doubles as the
        // snapshot: both element collections are EAGER, so the prior person ids
        // needed for cleanup are already loaded.
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

        // Phase B — network I/O with no transaction open.
        val outcome = runInference(photo)

        // Phase C — short write transaction.
        transactionTemplate.execute { writeOutcome(photoId, outcome) }
    }

    // Everything phase C needs to persist, produced without touching the
    // (detached) entity — mutations on it would be silently lost at the
    // phase-C reload.
    private class InferenceOutcome(
        val bytesUnavailable: String? = null,
        val facesOk: Boolean = false,
        val bibsOk: Boolean = false,
        val facesTransport: Boolean = false,
        val bibsTransport: Boolean = false,
        val personId: String? = null,
        val bibs: Map<String, BigDecimal> = emptyMap(),
        val error: String? = null,
    )

    private fun runInference(photo: Photo): InferenceOutcome {
        // Retry-safety: a re-run is a clean slate upstream too (no duplicate
        // ai-api persons). Best-effort — deleteFacesPerson is the same primitive
        // used for GDPR erasure, and a documented no-op under Rekognition.
        photo.facePersons.forEach { fp ->
            runCatching { aiApiClient.deleteFacesPerson(fp.aiPersonId) }
                .onFailure { log.warn("ai person cleanup failed for {}: {}", fp.aiPersonId, it.message) }
        }

        val bytes = try {
            storageService.getBytes(photo.s3Key)
        } catch (ex: Exception) {
            log.warn("Indexing {}: original bytes unavailable ({}); marking FAILED", photo.id, ex.message)
            return InferenceOutcome(bytesUnavailable = "original image unavailable: ${ex.message}")
        }
        // Index the ORIGINAL, never the watermarked image — the diagonal
        // anti-piracy overlay would corrupt face + bib detection.
        val contentType = sniffContentType(bytes)
        val filename = "${photo.id}.jpg"

        var error: String? = null

        // One enroll per photo: the photo IS the ai "person", and every face in
        // it (each a distinct runner) is stored under that single person_id, so
        // any of them matches at search time. NO_FACES / LOW_QUALITY are benign —
        // the call ran, there was simply nothing usable to enroll.
        var personId: String? = null
        var facesOk = false
        var facesTransport = false
        try {
            val enroll = aiApiClient.facesEnroll(
                file = bytes,
                contentType = contentType,
                filename = filename,
                personName = photo.id.toString(),
                personId = null,
                eventId = photo.eventId,
            )
            personId = enroll.person_id
            facesOk = true
        } catch (ex: AiApiException) {
            if (ex.aiCode == "NO_FACES" || ex.aiCode == "LOW_QUALITY") {
                facesOk = true
            } else {
                log.warn("Face enroll failed for photo {}: {}", photo.id, ex.message)
                error = "faces: ${ex.message}"
                facesTransport = ex.isRetryable
            }
        } catch (ex: Exception) {
            log.warn("Face enroll failed for photo {}: {}", photo.id, ex.message)
            error = "faces: ${ex.message}"
        }

        var bibs: Map<String, BigDecimal> = emptyMap()
        var bibsOk = false
        var bibsTransport = false
        try {
            val result = aiApiClient.bibsRecognize(bytes, contentType, filename)
            // Group only the QUALIFYING detections by normalized bib number and
            // store the max confidence among them — a re-lookup over the full
            // (unfiltered) list could otherwise persist a below-threshold value
            // when the same bib appears more than once in one photo.
            bibs = result.detections
                .filter { it.confidence >= aiApiProperties.bibConfidenceThresholdDefault }
                .groupBy { it.bib_number.trim().uppercase() }
                .filterKeys { it.isNotEmpty() }
                .mapValues { (_, detections) -> BigDecimal.valueOf(detections.maxOf { it.confidence }) }
            bibsOk = true
        } catch (ex: Exception) {
            log.warn("Bib recognize failed for photo {}: {}", photo.id, ex.message)
            error = listOfNotNull(error, "bibs: ${ex.message}").joinToString("; ")
            bibsTransport = (ex as? AiApiException)?.isRetryable == true
        }

        return InferenceOutcome(
            facesOk = facesOk,
            bibsOk = bibsOk,
            facesTransport = facesTransport,
            bibsTransport = bibsTransport,
            personId = personId,
            bibs = bibs,
            error = error,
        )
    }

    private fun writeOutcome(photoId: UUID, outcome: InferenceOutcome) {
        val photo = photoRepository.findById(photoId).orElse(null) ?: return
        if (photo.indexingStatus == IndexingStatus.INDEXED ||
            photo.indexingStatus == IndexingStatus.SKIPPED
        ) {
            // Another worker finished while we were in phase B — keep its
            // result, best-effort drop the person we just enrolled.
            outcome.personId?.let { pid ->
                runCatching { aiApiClient.deleteFacesPerson(pid) }
                    .onFailure { log.warn("ai person cleanup failed for {}: {}", pid, it.message) }
            }
            return
        }

        if (outcome.bytesUnavailable != null) {
            photo.facePersons.clear()
            photo.bibs.clear()
            photo.indexingStatus = IndexingStatus.FAILED
            photo.indexingAttempts += 1
            photo.indexingError = outcome.bytesUnavailable
            photo.indexedAt = null
            photoRepository.save(photo)
            return
        }

        if (!outcome.facesOk && !outcome.bibsOk && outcome.facesTransport && outcome.bibsTransport) {
            // Provider unreachable — both halves died at transport level. Keep
            // the attempt budget intact and fall back to PENDING so the
            // reconcile sweep re-drives it once the provider is back (mirrors
            // batch mode, where a failed drain rolls back without consuming the
            // attempt). Existing rows — e.g. a prior PARTIAL's bibs — survive.
            photo.indexingStatus = IndexingStatus.PENDING
            photo.indexingError = outcome.error
            photoRepository.save(photo)
            log.warn("Indexing {}: provider unreachable; back to PENDING, attempt not consumed", photoId)
            return
        }

        photo.indexingAttempts += 1
        photo.facePersons.clear()
        photo.bibs.clear()
        outcome.personId?.let {
            photo.facePersons.add(PhotoFacePersonEmbed(faceIndex = 0, aiPersonId = it))
        }
        outcome.bibs.forEach { (bibNumber, confidence) ->
            photo.bibs.add(PhotoBibEmbed(bibNumber = bibNumber, ocrConfidence = confidence))
        }
        photo.indexingError = outcome.error
        photo.indexingStatus = when {
            outcome.facesOk && outcome.bibsOk -> IndexingStatus.INDEXED
            outcome.facesOk || outcome.bibsOk -> IndexingStatus.PARTIAL
            else -> IndexingStatus.FAILED
        }
        photo.indexedAt =
            if (photo.indexingStatus == IndexingStatus.INDEXED) OffsetDateTime.now() else null
        photo.indexedProvider = aiProperties.provider.name.lowercase()
        if (photo.indexingStatus == IndexingStatus.FAILED &&
            photo.indexingAttempts >= aiApiProperties.maxIndexingAttempts
        ) {
            log.warn(
                "Photo {} exhausted {} indexing attempts; terminal FAILED — " +
                    "admin POST /admin/events/{}/photos/reindex re-drives it",
                photoId,
                photo.indexingAttempts,
                photo.eventId,
            )
        }
        photoRepository.save(photo)

        // Notify the live gallery even on PARTIAL so the bib (if any) surfaces.
        // Published inside the phase-C transaction so AFTER_COMMIT fires.
        if (outcome.facesOk || outcome.bibsOk) {
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

    // ai-api validates by magic bytes, not Content-Type, but the multipart part
    // still needs a sensible media type. Sniff it; default to JPEG.
    private fun sniffContentType(bytes: ByteArray): String = when {
        bytes.size >= 3 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xD8.toByte() && bytes[2] == 0xFF.toByte() -> "image/jpeg"
        bytes.size >= 8 && bytes[0] == 0x89.toByte() && bytes[1] == 0x50.toByte() -> "image/png"
        bytes.size >= 12 && bytes[0] == 'R'.code.toByte() && bytes[3] == 'F'.code.toByte() -> "image/webp"
        else -> "image/jpeg"
    }
}
