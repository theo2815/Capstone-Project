package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IndexingMode
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.repository.PhotoRepository
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Async
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener
import java.time.OffsetDateTime

// Drives PhotoIndexingService through two entry points:
//   1. Hot path — AFTER_COMMIT of an upload, dispatched to the imageProcessing
//      pool, so a photo is indexed within seconds of being uploaded.
//   2. Reconciliation — a periodic sweep that re-drives PENDING/FAILED photos
//      the hot path missed (app crash mid-index, ai-api outage) until they
//      settle into INDEXED/PARTIAL or exhaust their attempts and stay FAILED.
@Component
class PhotoIndexingTrigger(
    private val indexingService: PhotoIndexingService,
    private val photoRepository: PhotoRepository,
    private val aiApiProperties: AiApiProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Async("imageProcessing")
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onPhotoUploaded(event: PhotoUploadedForIndexing) {
        // In BATCH mode the per-photo hot path stands down — photos accumulate as
        // PENDING and PhotoBatchIndexingScheduler drains them per event.
        if (aiApiProperties.indexingMode == IndexingMode.BATCH) return
        // Never let an indexing failure escape the async worker — the row stays
        // PENDING/FAILED and the reconcile sweep retries it.
        try {
            indexingService.index(event.photoId)
        } catch (ex: Exception) {
            log.warn("Async index failed for photo {}: {}", event.photoId, ex.message)
        }
    }

    @Scheduled(fixedDelayString = "\${app.ai-api.reconcile-interval-ms:60000}")
    fun reconcile() {
        if (!aiApiProperties.enabled) return
        // BATCH mode uses its own drain + poll-ingest loops, not the per-photo sweep.
        if (aiApiProperties.indexingMode == IndexingMode.BATCH) return
        // Grace window: skip photos the hot path is probably still handling, so
        // the sweep only picks up genuinely-stuck work.
        val cutoff = OffsetDateTime.now().minusSeconds(RECONCILE_GRACE_SECONDS)
        val backlog = photoRepository.findIndexingBacklog(
            statuses = RETRYABLE_STATUSES,
            maxAttempts = aiApiProperties.maxIndexingAttempts,
            cutoff = cutoff,
            pageable = PageRequest.of(0, RECONCILE_BATCH_SIZE),
        )
        if (backlog.isEmpty()) return
        log.info("Reconciling {} stuck photo(s) for indexing", backlog.size)
        backlog.forEach { photo ->
            try {
                indexingService.index(photo.id)
            } catch (ex: Exception) {
                log.warn("Reconcile index failed for photo {}: {}", photo.id, ex.message)
            }
        }
    }

    private companion object {
        // PARTIAL is retryable: it only arises when one of face/bib hit a
        // *transient* ai-api error (the other succeeded), so re-driving the
        // idempotent index() can recover the missing half. Without this a brief
        // ai-api blip would strand that half forever, defeating the self-healing
        // goal. The V23 partial index covers all three so the sweep stays cheap.
        val RETRYABLE_STATUSES =
            listOf(IndexingStatus.PENDING, IndexingStatus.FAILED, IndexingStatus.PARTIAL)
        const val RECONCILE_GRACE_SECONDS = 120L
        const val RECONCILE_BATCH_SIZE = 200 // see PhotoWatermarkTrigger
    }
}
