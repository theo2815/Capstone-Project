package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IndexingMode
import com.quickpitik.entity.IndexJobStatus
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.repository.AiIndexJobRepository
import com.quickpitik.repository.PhotoRepository
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component
import java.time.OffsetDateTime

// Two single-threaded scheduled loops for Phase C batch mode (both no-op unless
// app.ai-api.indexing-mode=batch):
//   drain()        — group PENDING photos per event into mega jobs.
//   pollOpenJobs() — the durable ingest backstop: poll SUBMITTED jobs for results
//                    (this is the PRIMARY ingest path in dev, where ai-api's SSRF
//                    guard blocks registering a localhost webhook).
@Component
class PhotoBatchIndexingScheduler(
    private val batchIndexingService: PhotoBatchIndexingService,
    private val ingestService: PhotoBatchIngestService,
    private val photoRepository: PhotoRepository,
    private val jobRepository: AiIndexJobRepository,
    private val aiApiProperties: AiApiProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Scheduled(fixedDelayString = "\${app.ai-api.batch.drain-interval-ms:30000}")
    fun drain() {
        if (!batchModeActive()) return
        val cutoff = OffsetDateTime.now().minusSeconds(aiApiProperties.batch.drainGraceSeconds)
        val events = photoRepository.findEventsWithIndexingBacklog(
            RETRYABLE_STATUSES,
            aiApiProperties.maxIndexingAttempts,
            cutoff,
            PageRequest.of(0, aiApiProperties.batch.maxEventsPerTick),
        )
        if (events.isEmpty()) return
        log.info("Batch drain: {} event(s) with indexing backlog", events.size)
        events.forEach { eventId ->
            try {
                batchIndexingService.drainEvent(eventId, cutoff)
            } catch (ex: Exception) {
                log.warn("Batch drain failed for event {}: {}", eventId, ex.message)
            }
        }
    }

    @Scheduled(fixedDelayString = "\${app.ai-api.webhook.poll-interval-ms:30000}")
    fun pollOpenJobs() {
        if (!batchModeActive()) return
        val cutoff = OffsetDateTime.now().minusSeconds(aiApiProperties.webhook.pollGraceSeconds)
        val open = jobRepository.findOpenSubmittedBefore(
            IndexJobStatus.SUBMITTED,
            cutoff,
            PageRequest.of(0, POLL_BATCH_SIZE),
        )
        if (open.isEmpty()) return
        open.forEach { job ->
            try {
                ingestService.ingest(job.aiJobId)
            } catch (ex: Exception) {
                log.warn("Poll ingest failed for job {}: {}", job.aiJobId, ex.message)
            }
        }
    }

    private fun batchModeActive(): Boolean =
        aiApiProperties.enabled && aiApiProperties.indexingMode == IndexingMode.BATCH

    private companion object {
        val RETRYABLE_STATUSES = listOf(IndexingStatus.PENDING, IndexingStatus.FAILED)
        const val POLL_BATCH_SIZE = 25
    }
}
