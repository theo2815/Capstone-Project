package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.PermanentDeleteResponse
import com.quickpitik.entity.Photo
import com.quickpitik.dto.photos.*
import com.quickpitik.entity.RemovalActorType
import com.quickpitik.service.photos.PhotoErasureService
import com.quickpitik.entity.RemovalState
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.orders.PaymongoCheckoutReconciler
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
    private val eventRepository: EventRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val checkoutReconciler: PaymongoCheckoutReconciler,
    private val erasure: PhotoErasureService,
    private val accessGuard: PhotographerAccessGuard,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // Soft-delete: stamp deleted_at on the caller's still-active photos and
    // drop the event's active photo count by however many actually moved.
    // Returns the number affected (already-trashed ids are no-ops).
    @Transactional
    fun trash(photographerId: UUID, eventId: UUID, photoIds: List<UUID>): Int {
        requireNotRemovedEvent(eventId)
        if (photoIds.isEmpty()) return 0
        photoRepository.lockIdsForUpdate(photoIds.distinct().sorted())
        val affected = photoRepository.trashOwned(photographerId, eventId, photoIds, OffsetDateTime.now())
        if (affected > 0) adjustPhotoCount(eventId, photographerId, -affected)
        return affected
    }

    // Inverse of trash: clear deleted_at on the caller's trashed photos and
    // restore the event's active photo count. The photo returns with its
    // lifecycle status (LIVE/PROCESSING/HIDDEN) untouched.
    @Transactional
    fun restore(photographerId: UUID, eventId: UUID, photoIds: List<UUID>): Int {
        // Restoring republishes photos, so it needs the same approval as upload
        // (security audit 2026-10-14, D6); trash and permanent delete stay open.
        accessGuard.requireApproved(photographerId, "restoring photos")
        requireNotRemovedEvent(eventId)
        if (photoIds.isEmpty()) return 0
        photoRepository.findAllByIdForUpdate(photoIds.distinct().sorted())
            .filter { it.photographerId == photographerId && it.eventId == eventId }
            .forEach(::requireNoRemovalCase)
        val affected = photoRepository.restoreOwned(photographerId, eventId, photoIds)
        if (affected > 0) adjustPhotoCount(eventId, photographerId, affected)
        return affected
    }

    // Irreversible. Only trashed photos are eligible; a photo that has been
    // sold is refused (a buyer may still hold a download grant for it) and
    // reported in blockedSold rather than deleted.
    //
    // Commit an irreversible claim BEFORE storage I/O. A failed cleanup stays
    // retryable, but cannot masquerade as recoverable trash. No transaction is
    // held during provider or storage calls.
    fun permanentDelete(photographerId: UUID, eventId: UUID, photoIds: List<UUID>): PermanentDeleteResponse {
        requireNotRemovedEvent(eventId)
        val trashed = load(photographerId, eventId, photoIds).filter { it.deletedAt != null }
        trashed.forEach(::requireNoRemovalCase)
        if (trashed.isEmpty()) return PermanentDeleteResponse(deleted = emptyList(), blockedSold = emptyList())

        val deleted = mutableListOf<UUID>()
        val blocked = mutableListOf<UUID>()
        val pending = mutableListOf<UUID>()
        val failed = mutableListOf<UUID>()
        trashed.forEach { photo ->
            try {
                if (isSold(photo.id)) {
                    blocked += photo.id
                    return@forEach
                }
                if (!checkoutReconciler.releaseExpiredPhotoReservations(photo.id)) {
                    if (isSold(photo.id)) blocked += photo.id else pending += photo.id
                    return@forEach
                }
                val jobId = erasure.request(PhotoErasureCommand(photo.id, PhotoErasurePurpose.PHOTOGRAPHER_DELETE,
                    RemovalActorType.PHOTOGRAPHER, photographerId, "Photographer permanent deletion", commandId = UUID.randomUUID()))
                if (erasure.attempt(jobId) == PhotoErasureOutcome.COMPLETE) deleted += photo.id
                else failed += photo.id
            } catch (ex: Exception) {
                failed += photo.id
                log.warn("Permanent delete failed for photo {} (event {}): {}", photo.id, eventId, ex.javaClass.simpleName)
            }
        }
        return PermanentDeleteResponse(deleted = deleted, blockedSold = blocked, blockedPending = pending, failed = failed)
    }

    private fun isSold(photoId: UUID): Boolean =
        downloadGrantRepository.existsByIdPhotoId(photoId) ||
            orderItemRepository.countSalesByPhotoIds(listOf(photoId)).any { it.salesCount > 0 }

    private fun load(photographerId: UUID, eventId: UUID, photoIds: List<UUID>) =
        if (photoIds.isEmpty()) {
            emptyList()
        } else {
            photoRepository.findAllByIdInAndPhotographerIdAndEventId(photoIds, photographerId, eventId)
        }

    private fun requireNotRemovedEvent(eventId: UUID) {
        if (eventRepository.findById(eventId).orElse(null)?.deletedAt != null) {
            throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        }
    }

    private fun requireNoRemovalCase(photo: Photo) {
        if (photo.removalState != RemovalState.NONE) {
            throw ConflictException(code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "This photo is restricted by a removal case. Contact support for its case decision.")
        }
    }

    private fun adjustPhotoCount(eventId: UUID, photographerId: UUID, delta: Int) {
        // Match the upload transaction's lock order.
        eventPhotographerRepository.adjustPhotoCount(eventId, photographerId, delta)
        eventRepository.adjustPhotoCount(eventId, delta)
    }
}
