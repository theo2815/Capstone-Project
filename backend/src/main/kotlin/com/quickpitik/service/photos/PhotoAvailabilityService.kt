package com.quickpitik.service.photos

import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoRemovalItemRepository
import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.RemovalDecision
import com.quickpitik.entity.RemovalItemState
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

@Service
@Transactional(propagation = Propagation.MANDATORY)
class PhotoAvailabilityService(
    private val photos: PhotoRepository,
    private val items: PhotoRemovalItemRepository,
) {
    fun refreshRemovalState(photoIds: Set<UUID>) {
        if (photoIds.isEmpty()) return
        val ids = photoIds.sorted()
        // Shares checkout's photo lock order. A restriction committed first
        // must be visible before checkout can insert its reservation rows.
        photos.findAllByIdForUpdate(ids)
        photos.refreshRemovalState(ids)
    }

    fun authorizeRemoval(photoId: UUID, itemId: UUID, actorId: UUID) {
        if (photos.findAllByIdForUpdate(listOf(photoId)).isEmpty()) {
            throw NotFoundException(code = ErrorCodes.PHOTO_NOT_FOUND, message = "Photo not found")
        }
        val item = items.findById(itemId).orElse(null)
        if (item == null || item.photoId != photoId || item.photoIdSnapshot != photoId ||
            item.state !in setOf(RemovalItemState.REMOVAL_AUTHORIZED, RemovalItemState.REMOVED) ||
            item.adminDecision != RemovalDecision.REMOVE || item.adminDecisionActorId != actorId) {
            throw ConflictException(code = ErrorCodes.INVALID_STATE_TRANSITION, message = "Photo has no matching removal authorization")
        }
        photos.authorizeRemoval(photoId)
    }
}
