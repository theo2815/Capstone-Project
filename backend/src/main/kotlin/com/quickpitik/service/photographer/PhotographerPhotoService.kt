package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.PermanentDeleteResponse
import com.quickpitik.entity.Photo
import com.quickpitik.entity.RemovalState
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoUploadIntentRepository
import com.quickpitik.service.storage.StorageService
import com.quickpitik.service.orders.PaymongoCheckoutReconciler
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
import java.time.OffsetDateTime
import java.util.UUID

// Write-side of the photographer's photo library: trash (soft-delete), restore,
// and permanent delete. Ownership is enforced by scoping every load to
// (photographerId, eventId) — a foreign or wrong-event id simply isn't in the
// loaded set, so it's silently skipped (bulk) and can never be acted on.
@Service
class PhotographerPhotoService(
    private val photoRepository: PhotoRepository,
    private val photoUploadIntentRepository: PhotoUploadIntentRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val orderItemRepository: OrderItemRepository,
    private val storageService: StorageService,
    private val eventRepository: EventRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val checkoutReconciler: PaymongoCheckoutReconciler,
    private val transactionTemplate: TransactionTemplate,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // Soft-delete: stamp deleted_at on the caller's still-active photos and
    // drop the event's active photo count by however many actually moved.
    // Returns the number affected (already-trashed ids are no-ops).
    @Transactional
    fun trash(photographerId: UUID, eventId: UUID, photoIds: List<UUID>): Int {
        requireNotRemovedEvent(eventId)
        if (photoIds.isEmpty()) return 0
        val affected = photoRepository.trashOwned(photographerId, eventId, photoIds, OffsetDateTime.now())
        if (affected > 0) adjustPhotoCount(eventId, photographerId, -affected)
        return affected
    }

    // Inverse of trash: clear deleted_at on the caller's trashed photos and
    // restore the event's active photo count. The photo returns with its
    // lifecycle status (LIVE/PROCESSING/HIDDEN) untouched.
    @Transactional
    fun restore(photographerId: UUID, eventId: UUID, photoIds: List<UUID>): Int {
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
                val claimed = claimDeletion(photographerId, eventId, photo.id)
                if (claimed == null) {
                    failed += photo.id
                    return@forEach
                }
                // Include the deterministic derivative key even if a worker
                // stored it before publishing its database pointer.
                setOfNotNull(claimed.s3Key, claimed.watermarkS3Key, claimed.thumbnailS3Key,
                    "events/$eventId/photos/${photo.id}/watermark.jpg")
                    .forEach(storageService::delete)
                transactionTemplate.executeWithoutResult {
                    photoRepository.findAllByIdForUpdate(listOf(photo.id)).singleOrNull()?.let {
                        check(it.deletionStartedAt != null)
                        photoUploadIntentRepository.scheduleCleanupForResultPhoto(photo.id, OffsetDateTime.now())
                        photoRepository.delete(it)
                    }
                }
                deleted += photo.id
            } catch (ex: Exception) {
                failed += photo.id
                log.warn("Permanent delete failed for photo {} (event {}): {}", photo.id, eventId, ex.message)
            }
        }
        return PermanentDeleteResponse(deleted = deleted, blockedSold = blocked, blockedPending = pending, failed = failed)
    }

    private fun isSold(photoId: UUID): Boolean =
        downloadGrantRepository.existsByIdPhotoId(photoId) ||
            orderItemRepository.countSalesByPhotoIds(listOf(photoId)).any { it.salesCount > 0 }

    private fun claimDeletion(photographerId: UUID, eventId: UUID, photoId: UUID): Photo? =
        transactionTemplate.execute {
            // Checkout locks these same photo rows before reserving any order
            // items. Restore must also acquire the row lock through its UPDATE.
            val current = photoRepository.findAllByIdForUpdate(listOf(photoId)).singleOrNull()
                ?.takeIf { it.photographerId == photographerId && it.eventId == eventId && it.deletedAt != null }
                ?: return@execute null
            requireNoRemovalCase(current)
            if (current.watermarkWorkToken != null || current.indexingWorkToken != null) return@execute null
            if (photoRepository.hasUnfinishedWork(photoId, "ANY")) return@execute null
            if (orderItemRepository.existsByIdPhotoId(photoId) || downloadGrantRepository.existsByIdPhotoId(photoId)) {
                return@execute null
            }
            if (current.deletionStartedAt == null) photoRepository.startDeletion(photoId, OffsetDateTime.now())
            current
        }

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
