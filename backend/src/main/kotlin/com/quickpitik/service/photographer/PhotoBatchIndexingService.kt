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
import org.springframework.transaction.support.TransactionTemplate
import com.quickpitik.service.photos.PhotoWorkFence
import com.quickpitik.service.photos.PhotoWorkKind
import com.quickpitik.service.photos.PhotoWorkClaim
import com.quickpitik.entity.RemovalState
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
    private val transactionTemplate: TransactionTemplate,
    private val workFence: PhotoWorkFence,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // Persist claims and batch membership before any creating POST. Network I/O
    // has no open transaction; uncertain submissions retain their durable claims.
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
        // batch. This short transaction commits before any mega submission.
        val files = mutableListOf<NamedFile>()
        val batched = mutableListOf<Photo>()
        val claims = mutableListOf<PhotoWorkClaim>()
        for (photo in photos) {
            if (photo.removalState != RemovalState.NONE || photo.deletedAt != null) continue
            val claim = workFence.claim(photo.id, PhotoWorkKind.INDEXING) ?: continue
            val bytes = try {
                storageService.getBytes(photo.s3Key)
            } catch (ex: Exception) {
                transactionTemplate.executeWithoutResult {
                    if (workFence.isCurrent(claim)) {
                        val current = photoRepository.findById(photo.id).orElseThrow()
                        current.indexingStatus = IndexingStatus.FAILED
                        current.indexingAttempts += 1
                        current.indexingError = "original image unavailable: ${ex.message}"
                        photoRepository.save(current)
                    }
                }
                workFence.finish(claim)
                continue
            }
            files.add(NamedFile(photo.id.toString(), bytes, sniffContentType(bytes)))
            batched.add(photo)
            claims.add(claim)
        }
        if (batched.isEmpty()) return false

        val batch = AiIndexBatch(
            eventId = eventId,
            photoIds = batched.map { it.id }.toMutableList(),
        )
        // Prior cleanup is also external I/O; its targets were saved by claim().
        batched.forEach(::clearPriorResults)
        val prepared = transactionTemplate.execute {
            // Canonical UUID text has PostgreSQL's unsigned byte order;
            // UUID.compareTo uses signed longs and can invert shared lock order.
            if (!claims.sortedBy { it.photoId.toString() }.all(workFence::isCurrent)) return@execute false
            batchRepository.save(batch)
            claims.forEach { workFence.attributeBatch(it, batch.id) }
            val current = photoRepository.findAllById(batched.map { it.id })
            current.forEach { photo ->
                photo.facePersons.clear()
                photo.bibs.clear()
                photo.indexingStatus = IndexingStatus.BATCHING
                photo.indexingAttempts += 1
                photo.indexingError = null
            }
            photoRepository.saveAll(current)
            true
        } == true
        if (!prepared) {
            claims.forEach(workFence::finish)
            return false
        }
        try {
            val bibJobId = aiApiClient.bibsRecognizeMega(files).job_id
            transactionTemplate.executeWithoutResult {
                jobRepository.save(AiIndexJob(batchId = batch.id, kind = IndexJobKind.BIB, aiJobId = bibJobId))
                claims.forEach { workFence.recordJob(it, "BIB", bibJobId) }
            }
            val faceJobId = aiApiClient.facesEnrollMega(files, eventId).job_id
            transactionTemplate.executeWithoutResult {
                jobRepository.save(AiIndexJob(batchId = batch.id, kind = IndexJobKind.FACE, aiJobId = faceJobId))
                claims.forEach { workFence.recordJob(it, "FACE", faceJobId) }
            }
        } catch (ex: Exception) {
            // Preserve the transport retry budget, but never release the claim:
            // a lost response can mean the remote job is still running.
            transactionTemplate.executeWithoutResult {
                batched.sortedBy { it.id.toString() }.forEach { photo ->
                    val claim = claims.first { it.photoId == photo.id }
                    if (workFence.isCurrent(claim)) {
                        val current = photoRepository.findById(photo.id).orElseThrow()
                        current.indexingStatus = IndexingStatus.PENDING
                        current.indexingAttempts -= 1
                        photoRepository.save(current)
                    }
                }
            }
            throw ex
        }
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
