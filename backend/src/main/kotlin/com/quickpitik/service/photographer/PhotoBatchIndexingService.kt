package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.entity.AiIndexBatch
import com.quickpitik.entity.AiIndexJob
import com.quickpitik.entity.IndexJobKind
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.repository.AiIndexBatchRepository
import com.quickpitik.repository.AiIndexJobRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.NamedFile
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

// Phase C batch drain: gather one event's PENDING/FAILED photos and ship them to
// ai-api as a single FACE mega job + a single BIB mega job. Results are ingested
// later by PhotoBatchIngestService (via webhook in prod, poll in dev/backstop).
@Service
class PhotoBatchIndexingService(
    private val photoRepository: PhotoRepository,
    private val batchRepository: AiIndexBatchRepository,
    private val jobRepository: AiIndexJobRepository,
    private val storageService: StorageService,
    private val aiApiClient: AiApiClient,
    private val aiApiProperties: AiApiProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // All-in-one transaction. The @Scheduled drain is single-threaded so only one
    // drainEvent runs at a time — holding the tx across the two mega uploads
    // (≤ batch.max-size images) is bounded and contention-free. If either POST
    // throws (ai-api down) the whole tx rolls back: photos stay PENDING and the
    // attempt isn't consumed, so an outage self-heals on the next tick.
    @Transactional
    fun drainEvent(eventId: UUID, cutoff: OffsetDateTime): Boolean {
        val photos = photoRepository.findIndexingBacklogForEvent(
            eventId,
            RETRYABLE_STATUSES,
            aiApiProperties.maxIndexingAttempts,
            cutoff,
            PageRequest.of(0, aiApiProperties.batch.maxSize),
        )
        if (photos.isEmpty()) return false

        // Read ORIGINAL bytes (never the watermarked image — the overlay would
        // corrupt face + bib detection). A photo whose bytes are gone is a poison
        // row: mark it FAILED individually so it can't block the whole event's
        // batch. (If the megas below throw, this mark rolls back too and the row
        // is simply re-evaluated next tick.)
        val files = mutableListOf<NamedFile>()
        val batched = mutableListOf<Photo>()
        for (photo in photos) {
            val bytes = try {
                storageService.getBytes(photo.s3Key)
            } catch (ex: Exception) {
                photo.indexingStatus = IndexingStatus.FAILED
                photo.indexingAttempts += 1
                photo.indexingError = "original image unavailable: ${ex.message}"
                photoRepository.save(photo)
                continue
            }
            files.add(NamedFile(photo.id.toString(), bytes, sniffContentType(bytes)))
            batched.add(photo)
        }
        if (batched.isEmpty()) return false

        // Submit bib FIRST: it creates no persons, so a face-POST failure after it
        // orphans only a harmless auto-expiring job, never face embeddings.
        val bibJobId = aiApiClient.bibsRecognizeMega(files).job_id
        val faceJobId = aiApiClient.facesEnrollMega(files, eventId).job_id

        val batch = AiIndexBatch(
            eventId = eventId,
            photoIds = batched.map { it.id }.toMutableList(),
        )
        batchRepository.save(batch)
        jobRepository.save(AiIndexJob(batchId = batch.id, kind = IndexJobKind.BIB, aiJobId = bibJobId))
        jobRepository.save(AiIndexJob(batchId = batch.id, kind = IndexJobKind.FACE, aiJobId = faceJobId))

        batched.forEach { photo ->
            clearPriorResults(photo)
            photo.indexingStatus = IndexingStatus.BATCHING
            photo.indexingAttempts += 1
            photo.indexingError = null
        }
        photoRepository.saveAll(batched)
        log.info("Submitted indexing batch {} for event {} ({} photos)", batch.id, eventId, batched.size)
        return true
    }

    // Drops any ai person + tags from a prior attempt so a re-drained photo is a
    // clean slate. For a fresh PENDING photo this is a no-op (no facePersons yet).
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

    private companion object {
        val RETRYABLE_STATUSES = listOf(IndexingStatus.PENDING, IndexingStatus.FAILED)
    }
}
