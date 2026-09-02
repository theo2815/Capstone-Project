package com.quickpitik.service.photographer

import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.image.PerceptualHash
import com.quickpitik.service.storage.StorageService
import com.quickpitik.websocket.PhotoPublishedEvent
import io.micrometer.core.instrument.MeterRegistry
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Value
import org.springframework.context.ApplicationEventPublisher
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.io.ByteArrayInputStream
import java.util.UUID
import javax.imageio.ImageIO

// Generates the watermarked derivative for one uploaded photo and flips it
// PROCESSING → LIVE. Runs OFF the upload request (PhotoWatermarkTrigger), so
// the upload's 200 only guarantees the original is durably stored — the photo
// becomes runner-visible here, and the photo.published WebSocket frame fires
// here, once the watermark it references actually exists.
//
// Attempt accounting mirrors indexing's 2026-08-27 hardening: only SEMANTIC
// failures (undecodable bytes — retrying can never succeed) consume the
// processing_attempts budget; transport failures (storage/logo unreadable)
// leave it intact so the reconcile sweep keeps re-driving until the
// dependency recovers (e.g. the photographer re-uploads their watermark).
@Service
class PhotoWatermarkService(
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val photoRepository: PhotoRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
    private val watermarkService: WatermarkService,
    private val watermarkLogoCache: WatermarkLogoCache,
    private val eventPublisher: ApplicationEventPublisher,
    private val transactionTemplate: TransactionTemplate,
    private val meterRegistry: MeterRegistry,
    @Value("\${app.watermark.max-attempts:5}") private val maxAttempts: Int,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // Deliberately NOT @Transactional: the storage GET, decode + composite, and
    // the watermark PUT take seconds and must not pin a Hikari connection —
    // same rule as PhotoUploadService.upload. Only the flip runs in a
    // transaction so the WS publish fires AFTER_COMMIT of the visible state.
    fun process(photoId: UUID) {
        val photo = photoRepository.findById(photoId).orElse(null) ?: return
        if (photo.status != PhotoStatus.PROCESSING) return
        if (photo.processingAttempts >= maxAttempts) return
        val photographerId = photo.photographerId
        if (photographerId == null) {
            // Legacy/seed rows have no owner and therefore no watermark to
            // composite; they should never be PROCESSING. Semantic — burn.
            settleSemanticFailure(photo.id, photo.processingAttempts, "photo has no photographer")
            return
        }
        val settings = photographerSettingsRepository.findById(photographerId).orElse(null)
        val watermarkKeySetting = settings?.watermarkS3Key
        if (settings == null || watermarkKeySetting == null) {
            // Settings row or logo pointer missing. Transient from this job's
            // point of view — the photographer can (re)upload the logo and the
            // sweep will then succeed — so the budget stays intact.
            log.warn("Watermark logo not configured for photographer {}; photo {} stays PROCESSING", photographerId, photoId)
            meterRegistry.counter("qp.watermark.outcome", "outcome", "transport").increment()
            return
        }
        // The credit baked into the preview. photographer_id is FK-backed, so a
        // missing user row is an inconsistency, not a property of the bytes —
        // transport, like the missing logo.
        val user = userRepository.findById(photographerId).orElse(null)
        if (user == null) {
            log.warn("Photographer {} has no user row; photo {} stays PROCESSING", photographerId, photoId)
            meterRegistry.counter("qp.watermark.outcome", "outcome", "transport").increment()
            return
        }
        val credit = WatermarkCredit(
            name = settings.brandName?.takeIf { it.isNotBlank() } ?: user.name,
            handle = settings.handle,
            photoId = photo.id,
        )

        val marked = try {
            val original = storageService.getBytes(photo.s3Key)
            val logo = watermarkLogoCache.get(watermarkKeySetting)
            // Direct-to-storage uploads (2026-09-02) never pass through the
            // request-time pixel guard, so the decompression-bomb check has to
            // live here, before the only full decode of client bytes. Semantic:
            // the object never changes, so burn an attempt.
            if (com.quickpitik.service.image.ImagePixelGuard.exceedsPixelBudget(original)) {
                settleSemanticFailure(photo.id, photo.processingAttempts, "image exceeds the pixel budget")
                return
            }
            try {
                watermarkService.processThumbnail(original, logo, credit)
            } catch (ex: IllegalArgumentException) {
                settleSemanticFailure(photo.id, photo.processingAttempts, ex.message ?: "undecodable image")
                return
            } catch (ex: java.io.IOException) {
                // ImageIO signals truncated/corrupt bytes with IOException — the
                // bytes in storage never change, so retrying can never succeed.
                settleSemanticFailure(photo.id, photo.processingAttempts, ex.message ?: "undecodable image")
                return
            }
        } catch (ex: Exception) {
            // Storage GET / logo fetch failed — transport. Sweep retries later.
            log.warn("Watermark transport failure for photo {}: {}", photoId, ex.message)
            meterRegistry.counter("qp.watermark.outcome", "outcome", "transport").increment()
            return
        }

        val watermarkKey = "events/${photo.eventId}/photos/${photo.id}/watermark.jpg"
        try {
            storageService.put(watermarkKey, marked.jpeg, "image/jpeg")
        } catch (ex: Exception) {
            log.warn("Watermark PUT failed for photo {}: {}", photoId, ex.message)
            meterRegistry.counter("qp.watermark.outcome", "outcome", "transport").increment()
            return
        }

        // Presign outside the transaction — local SigV4 math, no storage call.
        val thumbnailUrl =
            storageService.presignedGetUrl(watermarkKey, storageProperties.presignedTtl.thumbnail)

        transactionTemplate.execute {
            val flipped = photoRepository.publishWatermarked(photo.id, watermarkKey, marked.phash)
            if (flipped == 1) {
                // Same frame shape the upload transaction used to publish —
                // PhotoPublishedBroadcaster (AFTER_COMMIT) is untouched. Fired
                // only now, when the imageUrl it carries actually resolves.
                eventPublisher.publishEvent(
                    PhotoPublishedEvent(
                        eventId = photo.eventId,
                        payload = mapOf(
                            "type" to "photo.published",
                            "photo" to mapOf(
                                "id" to photo.id.toString(),
                                "bib" to (photo.bibs.minByOrNull { it.bibNumber }?.bibNumber),
                                "tone" to photo.tone,
                                "span" to photo.span.wire,
                                "imageUrl" to thumbnailUrl,
                                "uploadedAt" to photo.uploadedAt.toString(),
                            ),
                        ),
                    ),
                )
            }
        }
        meterRegistry.counter("qp.watermark.outcome", "outcome", "live").increment()
    }

    // Registers the fingerprint for a LIVE preview that predates the phash
    // column (V42). Driven by PhotoWatermarkTrigger.backfillPhash in bounded
    // batches; any failure just leaves the row for the next sweep.
    // ponytail: a row whose watermark object is gone retries every sweep; bounded by the batch.
    fun backfillPhash(photo: Photo) {
        val key = photo.watermarkS3Key ?: return
        try {
            val preview = ImageIO.read(ByteArrayInputStream(storageService.getBytes(key)))
            if (preview == null) {
                log.warn("Watermark object {} for photo {} is not decodable; phash left null", key, photo.id)
                return
            }
            val phash = PerceptualHash.of(preview)
            transactionTemplate.execute { photoRepository.setPhash(photo.id, phash) }
        } catch (ex: Exception) {
            log.warn("phash backfill failed for photo {}: {}", photo.id, ex.message)
        }
    }

    private fun settleSemanticFailure(photoId: UUID, attemptsBefore: Int, reason: String) {
        transactionTemplate.execute { photoRepository.incrementProcessingAttempts(photoId) }
        meterRegistry.counter("qp.watermark.outcome", "outcome", "failed").increment()
        if (attemptsBefore + 1 >= maxAttempts) {
            log.warn(
                "Photo {} exhausted {} watermark attempts ({}); stays PROCESSING and invisible to runners",
                photoId,
                attemptsBefore + 1,
                reason,
            )
        } else {
            log.warn("Watermark failed for photo {} (attempt {}): {}", photoId, attemptsBefore + 1, reason)
        }
    }
}
