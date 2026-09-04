package com.quickpitik.service.photographer

import com.quickpitik.repository.PhotoRepository
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Value
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Async
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener
import java.time.OffsetDateTime

// Drives PhotoWatermarkService through two entry points — the same shape as
// PhotoIndexingTrigger:
//   1. Hot path — AFTER_COMMIT of an upload, on the imageProcessing pool, so a
//      photo flips PROCESSING → LIVE within a second of being uploaded.
//   2. Reconciliation — a periodic sweep that re-drives PROCESSING photos the
//      hot path missed (app crash mid-watermark, storage outage) until they go
//      LIVE or exhaust their semantic-failure budget.
@Component
class PhotoWatermarkTrigger(
    private val watermarkService: PhotoWatermarkService,
    private val photoRepository: PhotoRepository,
    @Value("\${app.watermark.max-attempts:5}") private val maxAttempts: Int,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Async("watermarkProcessing")
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onPhotoUploaded(event: PhotoUploadedForWatermark) {
        // Never let a watermark failure escape the async worker — the row stays
        // PROCESSING and the reconcile sweep retries it.
        try {
            watermarkService.process(event.photoId)
        } catch (ex: Exception) {
            log.warn("Async watermark failed for photo {}: {}", event.photoId, ex.message)
        }
    }

    @Scheduled(fixedDelayString = "\${app.watermark.reconcile-interval-ms:60000}")
    fun reconcile() {
        // Grace window: skip photos the hot path is probably still handling.
        val cutoff = OffsetDateTime.now().minusSeconds(RECONCILE_GRACE_SECONDS)
        val backlog = photoRepository.findWatermarkBacklog(
            maxAttempts = maxAttempts,
            cutoff = cutoff,
            pageable = PageRequest.of(0, RECONCILE_BATCH_SIZE),
        )
        if (backlog.isEmpty()) return
        log.info("Reconciling {} stuck photo(s) for watermarking", backlog.size)
        backlog.forEach { photo ->
            try {
                watermarkService.process(photo.id)
            } catch (ex: Exception) {
                log.warn("Reconcile watermark failed for photo {}: {}", photo.id, ex.message)
            }
        }
    }

    // Fingerprint registry catch-up (V42): LIVE previews that predate the
    // phash column get hashed here, a bounded batch per tick.
    @Scheduled(fixedDelayString = "\${app.watermark.reconcile-interval-ms:60000}")
    fun backfillPhash() {
        val backlog = photoRepository.findPhashBacklog(PageRequest.of(0, PHASH_BATCH_SIZE))
        if (backlog.isEmpty()) return
        log.info("Backfilling phash for {} live photo(s)", backlog.size)
        backlog.forEach { watermarkService.backfillPhash(it) }
    }

    private companion object {
        const val RECONCILE_GRACE_SECONDS = 120L
        const val PHASH_BATCH_SIZE = 50
        // 25 capped recovery at 25 photos/min — a 1,000-frame backlog took 40
        // minutes to surface. 200/min covers a burst; the pool bounds the rest.
        const val RECONCILE_BATCH_SIZE = 200
    }
}
