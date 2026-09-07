package com.quickpitik.service.photographer

import com.quickpitik.dto.photographer.PermanentDeleteResponse
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

// Write-side of the photographer's photo library: trash (soft-delete), restore,
// and permanent delete. Ownership is enforced by scoping every load to
// (photographerId, eventId) — a foreign or wrong-event id simply isn't in the
// loaded set, so it's silently skipped (bulk) and can never be acted on.
@Service
class PhotographerPhotoService(
    private val photoRepository: PhotoRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val orderItemRepository: OrderItemRepository,
    private val storageService: StorageService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // Soft-delete: stamp deleted_at on the caller's still-active photos and
    // drop the event's active photo count by however many actually moved.
    // Returns the number affected (already-trashed ids are no-ops).
    @Transactional
    fun trash(photographerId: UUID, eventId: UUID, photoIds: List<UUID>): Int {
        val moving = load(photographerId, eventId, photoIds).filter { it.deletedAt == null }
        if (moving.isEmpty()) return 0
        val now = OffsetDateTime.now()
        moving.forEach { it.deletedAt = now }
        photoRepository.saveAll(moving)
        adjustPhotoCount(eventId, photographerId, -moving.size)
        return moving.size
    }

    // Inverse of trash: clear deleted_at on the caller's trashed photos and
    // restore the event's active photo count. The photo returns with its
    // lifecycle status (LIVE/PROCESSING/HIDDEN) untouched.
    @Transactional
    fun restore(photographerId: UUID, eventId: UUID, photoIds: List<UUID>): Int {
        val moving = load(photographerId, eventId, photoIds).filter { it.deletedAt != null }
        if (moving.isEmpty()) return 0
        moving.forEach { it.deletedAt = null }
        photoRepository.saveAll(moving)
        adjustPhotoCount(eventId, photographerId, moving.size)
        return moving.size
    }

    // Irreversible. Only trashed photos are eligible; a photo that has been
    // sold is refused (a buyer may still hold a download grant for it) and
    // reported in blockedSold rather than deleted.
    //
    // Deliberately NOT @Transactional at the method level: R2 deletes are
    // external I/O and must not hold a DB transaction open across the network.
    // R2 first (idempotent — a missing key is a no-op on both backends), then
    // the row (its own transaction via the repository). A partial R2 failure
    // leaves the row trashed and retry-safe; no orphaned R2 object ever
    // survives a committed row delete.
    fun permanentDelete(photographerId: UUID, eventId: UUID, photoIds: List<UUID>): PermanentDeleteResponse {
        val trashed = load(photographerId, eventId, photoIds).filter { it.deletedAt != null }
        if (trashed.isEmpty()) return PermanentDeleteResponse(deleted = emptyList(), blockedSold = emptyList())

        val soldIds = orderItemRepository.countSalesByPhotoIds(trashed.map { it.id })
            .filter { it.salesCount > 0 }
            .mapTo(mutableSetOf()) { it.photoId }

        val deleted = mutableListOf<UUID>()
        val blocked = mutableListOf<UUID>()
        trashed.forEach { photo ->
            if (photo.id in soldIds) {
                blocked += photo.id
                return@forEach
            }
            try {
                // thumbnail_s3_key == watermark_s3_key in practice — setOfNotNull
                // dedups so the same object isn't deleted twice.
                setOfNotNull(photo.s3Key, photo.watermarkS3Key, photo.thumbnailS3Key)
                    .forEach(storageService::delete)
                photoRepository.delete(photo)
                deleted += photo.id
            } catch (ex: Exception) {
                // Leave the photo in trash; the next permanent-delete retry is
                // safe (R2 delete is idempotent). Never delete the row when its
                // objects might still exist.
                log.warn("Permanent delete failed for photo {} (event {}): {}", photo.id, eventId, ex.message)
            }
        }
        return PermanentDeleteResponse(deleted = deleted, blockedSold = blocked)
    }

    private fun load(photographerId: UUID, eventId: UUID, photoIds: List<UUID>) =
        if (photoIds.isEmpty()) {
            emptyList()
        } else {
            photoRepository.findAllByIdInAndPhotographerIdAndEventId(photoIds, photographerId, eventId)
        }

    private fun adjustPhotoCount(eventId: UUID, photographerId: UUID, delta: Int) {
        eventPhotographerRepository.findById(EventPhotographerId(eventId, photographerId)).orElse(null)?.let { ep ->
            ep.photoCount = (ep.photoCount + delta).coerceAtLeast(0)
            eventPhotographerRepository.save(ep)
        }
    }
}
