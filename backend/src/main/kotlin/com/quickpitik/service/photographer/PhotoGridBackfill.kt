package com.quickpitik.service.photographer

import com.quickpitik.config.ThumbnailProperties
import com.quickpitik.entity.Photo
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.photos.PhotoWorkFence
import com.quickpitik.service.photos.PhotoWorkKind
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component
import org.springframework.transaction.support.TransactionTemplate
import java.util.UUID

// Gives photos published before the 400 px grid tile (FS-9) their grid.jpg,
// cut from the already-marked preview. Off unless app.thumbnails.backfill-enabled.
// One pass per app start, walking forward by id: a photo that fails is logged
// and skipped, so it cannot stall the pass; a restart retries it.
@Component
class PhotoGridBackfill(
    private val photoRepository: PhotoRepository,
    private val storageService: StorageService,
    private val watermarkService: WatermarkService,
    private val workFence: PhotoWorkFence,
    private val transactionTemplate: TransactionTemplate,
    private val properties: ThumbnailProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)
    private var after = UUID(0, 0)
    private var written = 0
    private var skipped = 0
    private var done = false

    @Scheduled(fixedDelay = TICK_MS)
    fun backfill() {
        if (!properties.backfillEnabled || done) return
        val batch = photoRepository.findGridBackfill(after, PageRequest.of(0, BATCH_SIZE))
        if (batch.isEmpty()) {
            done = true
            log.info("thumbnail backfill complete: {} tile(s) written, {} photo(s) skipped (a restart retries them)", written, skipped)
            return
        }
        batch.forEach { if (backfillOne(it)) written++ else skipped++ }
        after = batch.last().id
    }

    // Same fence discipline as PhotoWatermarkService: the claim keeps erasure from
    // racing the PUT, and a PUT that may still land remotely keeps the claim held.
    private fun backfillOne(photo: Photo): Boolean {
        val claim = workFence.claim(photo.id, PhotoWorkKind.WATERMARK) ?: run {
            log.warn("Grid backfill {}: work claim not granted; skipped", photo.id)
            return false
        }
        // Erasure lists this fixed path in every manifest (PhotoErasureService).
        val gridKey = "events/${photo.eventId}/photos/${photo.id}/grid.jpg"
        var unresolvedWrite = false
        try {
            val tile = watermarkService.gridFromPreview(storageService.getBytes(photo.watermarkS3Key!!))
            if (!workFence.isCurrent(claim)) {
                log.warn("Grid backfill {}: work claim ended before the upload; skipped", photo.id)
                return false
            }
            unresolvedWrite = true
            storageService.put(gridKey, tile, "image/jpeg")
            unresolvedWrite = false
            return transactionTemplate.execute {
                workFence.isCurrent(claim) && photoRepository.setGridKey(photo.id, gridKey) == 1
            } == true
        } catch (ex: Exception) {
            log.warn("Grid backfill failed for photo {}: {}", photo.id, ex.message)
            return false
        } finally {
            if (!unresolvedWrite) workFence.finish(claim)
        }
    }

    private companion object {
        const val TICK_MS = 60_000L
        const val BATCH_SIZE = 50
    }
}
