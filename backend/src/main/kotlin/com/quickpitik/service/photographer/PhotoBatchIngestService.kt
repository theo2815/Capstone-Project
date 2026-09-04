package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.AiProperties
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
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.websocket.PhotoIndexedEvent
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

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
    private val aiProperties: AiProperties,
    private val eventPublisher: ApplicationEventPublisher,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional
    fun ingest(aiJobId: String) {
        val job = jobRepository.findByAiJobId(aiJobId) ?: return
        if (job.status != IndexJobStatus.SUBMITTED) return // idempotent — already processed
        // Lock the batch so the FACE and BIB ingests of the same batch serialize;
        // the second one finalizes once it sees the first's committed job status.
        val batch = batchRepository.findByIdForUpdate(job.batchId) ?: return

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
                job.status = IndexJobStatus.FAILED
                job.error = "ai-api job not found (expired or lost)"
                job.completedAt = OffsetDateTime.now()
                jobRepository.save(job)
                maybeFinalize(batch)
            } else {
                log.warn("ingest: jobStatus({}) failed: {} — leaving SUBMITTED", aiJobId, ex.message)
            }
            return
        } catch (ex: Exception) {
            // Transient ai-api error — leave SUBMITTED; the poll backstop retries.
            log.warn("ingest: jobStatus({}) failed: {} — leaving SUBMITTED", aiJobId, ex.message)
            return
        }

        when (status.status) {
            "completed" -> {
                writeResults(batch, job.kind, status.result ?: emptyList())
                job.status = IndexJobStatus.COMPLETED
                job.completedAt = OffsetDateTime.now()
            }
            "failed" -> {
                job.status = IndexJobStatus.FAILED
                job.error = status.error
                job.completedAt = OffsetDateTime.now()
            }
            else -> return // pending/processing — not terminal yet
        }
        jobRepository.save(job)
        maybeFinalize(batch)
    }

    // Finalize the batch once no job is still SUBMITTED — i.e. both the FACE and
    // BIB jobs have reached a terminal state (COMPLETED or FAILED).
    private fun maybeFinalize(batch: AiIndexBatch) {
        val jobs = jobRepository.findByBatchId(batch.id)
        if (jobs.none { it.status == IndexJobStatus.SUBMITTED }) {
            finalize(batch, jobs)
        }
    }

    private fun writeResults(batch: AiIndexBatch, kind: IndexJobKind, result: List<Any?>) {
        val photosById = photoRepository.findAllById(batch.photoIds).associateBy { it.id }
        result.forEachIndexed { idx, item ->
            val map = item as? Map<*, *> ?: return@forEachIndexed
            val photo = resolvePhoto(map, idx, batch.photoIds, photosById) ?: return@forEachIndexed
            if (photo.indexingStatus != IndexingStatus.BATCHING) return@forEachIndexed
            when (kind) {
                IndexJobKind.FACE -> {
                    // person_id is null for a photo with no enrollable face — benign.
                    val personId = (map["person_id"] as? String)?.takeIf { it.isNotBlank() }
                    if (personId != null && photo.facePersons.none { it.aiPersonId == personId }) {
                        photo.facePersons.add(PhotoFacePersonEmbed(faceIndex = 0, aiPersonId = personId))
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

    // Map a result item to its photo: prefer the echoed ref (= photo id), fall
    // back to the result's own global index, then the positional list index.
    private fun resolvePhoto(
        map: Map<*, *>,
        idx: Int,
        photoIds: List<UUID>,
        photosById: Map<UUID, Photo>,
    ): Photo? {
        val ref = (map["ref"] as? String)?.let { runCatching { UUID.fromString(it) }.getOrNull() }
        val pos = (map["index"] as? Number)?.toInt() ?: idx
        val id = ref ?: photoIds.getOrNull(pos)
        return id?.let { photosById[it] }
    }

    private fun finalize(batch: AiIndexBatch, jobs: List<AiIndexJob>) {
        val faceOk = jobs.any { it.kind == IndexJobKind.FACE && it.status == IndexJobStatus.COMPLETED }
        val bibOk = jobs.any { it.kind == IndexJobKind.BIB && it.status == IndexJobStatus.COMPLETED }
        val now = OffsetDateTime.now()
        val photos = photoRepository.findAllById(batch.photoIds)
        photos.forEach { photo ->
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
                photo.indexedProvider = aiProperties.provider.name.lowercase()
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
        batch.status = if (faceOk || bibOk) BatchStatus.FINALIZED else BatchStatus.FAILED
        batch.finalizedAt = now
        batchRepository.save(batch)
        log.info("Finalized batch {} (faceOk={}, bibOk={}, {} photos)", batch.id, faceOk, bibOk, photos.size)
    }
}
