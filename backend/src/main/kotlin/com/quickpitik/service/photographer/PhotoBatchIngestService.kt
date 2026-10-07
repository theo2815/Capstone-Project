package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.AiProvider
import com.quickpitik.entity.AiIndexBatch
import com.quickpitik.entity.AiIndexJob
import com.quickpitik.entity.BatchStatus
import com.quickpitik.entity.IndexJobKind
import com.quickpitik.entity.IndexJobStatus
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoBibEmbed
import com.quickpitik.entity.PhotoFacePersonEmbed
import com.quickpitik.repository.AiIndexBatchRepository
import com.quickpitik.repository.AiIndexJobRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.faceResultPhoto
import com.quickpitik.service.ai.jobResultPhoto
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.websocket.PhotoIndexedEvent
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import com.quickpitik.service.photos.PhotoWorkFence
import com.quickpitik.dto.ai.JobStatusResult
import com.quickpitik.entity.RemovalState
import com.quickpitik.entity.FaceProvider
import java.math.BigDecimal
import java.time.OffsetDateTime

// Ingests one mega job's results into the photos of its batch. Driven by the
// webhook receiver (prod) or the poll backstop (dev/always-on) — both call
// ingest(aiJobId). A photo settles (INDEXED/PARTIAL/FAILED) only once BOTH its
// jobs are terminal. Idempotent: an already-terminal job is a no-op.
@Service
class PhotoBatchIngestService(
    private val photoRepository: PhotoRepository,
    private val batchRepository: AiIndexBatchRepository,
    private val jobRepository: AiIndexJobRepository,
    private val aiApiClient: AiApiClient,
    private val aiApiProperties: AiApiProperties,
    private val eventPublisher: ApplicationEventPublisher,
    private val transactionTemplate: TransactionTemplate,
    private val workFence: PhotoWorkFence,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun ingest(aiJobId: String) {
        val job = jobRepository.findByAiJobId(aiJobId) ?: return
        if (job.status != IndexJobStatus.SUBMITTED) return // idempotent — already processed
        val batch = batchRepository.findById(job.batchId).orElse(null) ?: return

        val status = try {
            aiApiClient.jobStatus(aiJobId, offset = 0, limit = batch.photoIds.size.coerceIn(1, 500))
        } catch (ex: AiApiException) {
            // A gone job (expired past ai-api's JOB_RETENTION_DAYS, or otherwise
            // lost) is reported as a NOT_FOUND envelope, surfaced here as
            // aiCode == "NOT_FOUND". It can never produce results, so terminate it —
            // otherwise the poll backstop re-polls it forever and its photos stay
            // BATCHING (never re-drained, never searchable). Any other ai-api error
            // is treated as transient: leave SUBMITTED for the next poll.
            if (ex.aiCode == "NOT_FOUND") {
                log.warn("ingest: job {} not found on ai-api — marking FAILED", aiJobId)
                settle(aiJobId, JobStatusResult(job_id = aiJobId, status = "failed",
                    error = "ai-api job not found (expired or lost)"))
            } else {
                log.warn("ingest: jobStatus({}) failed: {} — leaving SUBMITTED", aiJobId, ex.message)
            }
            return
        } catch (ex: Exception) {
            // Transient ai-api error — leave SUBMITTED; the poll backstop retries.
            log.warn("ingest: jobStatus({}) failed: {} — leaving SUBMITTED", aiJobId, ex.message)
            return
        }

        settle(aiJobId, status)
    }

    private fun settle(aiJobId: String, status: JobStatusResult) = transactionTemplate.executeWithoutResult {
        val initial = jobRepository.findByAiJobId(aiJobId) ?: return@executeWithoutResult
        val batch = batchRepository.findByIdForUpdate(initial.batchId) ?: return@executeWithoutResult
        val job = jobRepository.findByAiJobId(aiJobId) ?: return@executeWithoutResult
        if (job.status != IndexJobStatus.SUBMITTED) return@executeWithoutResult
        when (status.status) {
            "completed" -> {
                if (job.kind == IndexJobKind.FACE && !completeFaceResults(batch, status.result)) {
                    // Keep observed late IDs without certifying the missing
                    // results or publishing a partially attributed completion.
                    writeResults(batch, job.kind, status.result ?: emptyList(), publish = false)
                    log.warn("ingest: face job {} has incomplete attribution; leaving SUBMITTED", aiJobId)
                    return@executeWithoutResult
                }
                writeResults(batch, job.kind, status.result ?: emptyList())
                job.status = IndexJobStatus.COMPLETED
                job.completedAt = OffsetDateTime.now()
            }
            "failed" -> {
                job.status = IndexJobStatus.FAILED
                job.error = status.error
                job.completedAt = OffsetDateTime.now()
            }
            "pending", "processing" -> {
                workFence.forBatch(batch.id).sortedBy { it.photoId.toString() }.forEach(workFence::renewBatchClaim)
                return@executeWithoutResult
            }
            else -> return@executeWithoutResult
        }
        jobRepository.save(job)
        maybeFinalize(batch)
    }

    // Finalize the batch once no job is still SUBMITTED — i.e. both the FACE and
    // BIB jobs have reached a terminal state (COMPLETED or FAILED).
    private fun maybeFinalize(batch: AiIndexBatch) {
        val jobs = jobRepository.findByBatchId(batch.id)
        if (jobs.map { it.kind }.toSet().size == 2 && jobs.none { it.status == IndexJobStatus.SUBMITTED }) {
            finalize(batch, jobs)
        }
    }

    private fun completeFaceResults(batch: AiIndexBatch, result: List<Any?>?): Boolean {
        if (result == null || result.size != batch.photoIds.size) return false
        val ids = result.mapIndexed { index, item ->
            faceResultPhoto(item, index, batch.photoIds) ?: return false
        }
        return ids.toSet() == batch.photoIds.toSet()
    }

    private fun writeResults(batch: AiIndexBatch, kind: IndexJobKind, result: List<Any?>, publish: Boolean = true) {
        val claims = workFence.forBatch(batch.id).associateBy { it.photoId }
        val current = claims.values.sortedBy { it.photoId.toString() }.associate { it.photoId to workFence.isCurrent(it) }
        val photosById = photoRepository.findAllById(batch.photoIds).associateBy { it.id }
        result.forEachIndexed { idx, item ->
            val map = item as? Map<*, *> ?: return@forEachIndexed
            val id = jobResultPhoto(map, idx, batch.photoIds) ?: return@forEachIndexed
            val claim = claims[id] ?: return@forEachIndexed
            val personId = (map["person_id"] as? String)?.takeIf { it.isNotBlank() }
            if (kind == IndexJobKind.FACE && personId != null) workFence.recordPerson(claim, personId)
            if (!publish) return@forEachIndexed
            val photo = photosById[id] ?: return@forEachIndexed
            if (current[id] != true || photo.removalState != RemovalState.NONE || photo.deletedAt != null) return@forEachIndexed
            if (photo.indexingStatus != IndexingStatus.BATCHING) return@forEachIndexed
            // These jobs are served by AiApiClient, even if the configured
            // provider changes before completion. Stamp partial results too.
            photo.indexedProvider = AiProvider.AI_API.name.lowercase()
            when (kind) {
                IndexJobKind.FACE -> {
                    // person_id is null for a photo with no enrollable face — benign.
                    if (personId != null && photo.facePersons.none { it.aiPersonId == personId }) {
                        photo.facePersons.add(PhotoFacePersonEmbed(faceIndex = 0, aiPersonId = personId, provider = FaceProvider.AI_API))
                    }
                }
                IndexJobKind.BIB -> {
                    val bibs = map["bibs"] as? List<*> ?: emptyList<Any?>()
                    bibs.forEach { b ->
                        val bm = b as? Map<*, *> ?: return@forEach
                        val conf = (bm["confidence"] as? Number)?.toDouble() ?: 0.0
                        val bibNumber = (bm["bib_number"] as? String)?.trim()?.uppercase()
                        if (!bibNumber.isNullOrEmpty() &&
                            conf >= aiApiProperties.bibConfidenceThresholdDefault &&
                            photo.bibs.none { it.bibNumber == bibNumber }
                        ) {
                            photo.bibs.add(PhotoBibEmbed(bibNumber, BigDecimal.valueOf(conf)))
                        }
                    }
                }
            }
        }
        photoRepository.saveAll(photosById.values)
    }

    private fun finalize(batch: AiIndexBatch, jobs: List<AiIndexJob>) {
        val faceOk = jobs.any { it.kind == IndexJobKind.FACE && it.status == IndexJobStatus.COMPLETED }
        val bibOk = jobs.any { it.kind == IndexJobKind.BIB && it.status == IndexJobStatus.COMPLETED }
        val now = OffsetDateTime.now()
        val claims = workFence.forBatch(batch.id).associateBy { it.photoId }
        val current = claims.values.sortedBy { it.photoId.toString() }.associate { it.photoId to workFence.isCurrent(it) }
        val photos = photoRepository.findAllById(batch.photoIds)
        photos.forEach { photo ->
            // Completed remote work may outlive its advisory local deadline.
            // Keep its targets, reject publication, and allow a fresh attempt
            // only when this is still the photo's own batch and it is available.
            if (current[photo.id] != true && faceOk && claims[photo.id]?.token == photo.indexingWorkToken &&
                photo.removalState == RemovalState.NONE && photo.deletedAt == null &&
                photo.deletionStartedAt == null && photo.indexingStatus == IndexingStatus.BATCHING) {
                photo.indexingStatus = IndexingStatus.PENDING
                photo.indexedAt = null
            }
            if (current[photo.id] != true || photo.removalState != RemovalState.NONE || photo.deletedAt != null) return@forEach
            if (photo.indexingStatus != IndexingStatus.BATCHING) return@forEach
            photo.indexingStatus = when {
                faceOk && bibOk -> IndexingStatus.INDEXED
                faceOk || bibOk -> IndexingStatus.PARTIAL
                photo.indexingAttempts >= aiApiProperties.maxIndexingAttempts -> IndexingStatus.FAILED
                else -> IndexingStatus.PENDING // both jobs failed → re-drain next tick
            }
            photo.indexedAt = if (photo.indexingStatus == IndexingStatus.INDEXED) now else null
            if (faceOk || bibOk) {
                // Stamp which provider's ID space the stored results belong to
                // (V33) — a later app.ai.provider flip makes these rows visibly
                // stale instead of silently unsearchable.
                photo.indexedProvider = AiProvider.AI_API.name.lowercase()
            }
            if (!faceOk && !bibOk) {
                // nothing usable was written; drop partials before re-drain/terminal
                photo.facePersons.clear()
                photo.bibs.clear()
            }
            // Notify the live gallery even on PARTIAL so the bib (if any) surfaces.
            if (faceOk || bibOk) {
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
        photoRepository.saveAll(photos)
        // Failed/lost face jobs may have produced partial output. Only a
        // completed FACE result proves the writer ended with attributable results.
        if (faceOk) claims.values.forEach(workFence::finish)
        batch.status = if (faceOk || bibOk) BatchStatus.FINALIZED else BatchStatus.FAILED
        batch.finalizedAt = now
        batchRepository.save(batch)
        log.info("Finalized batch {} (faceOk={}, bibOk={}, {} photos)", batch.id, faceOk, bibOk, photos.size)
    }
}
