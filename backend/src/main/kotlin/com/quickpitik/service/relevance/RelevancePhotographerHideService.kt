package com.quickpitik.service.relevance

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.PhotoActionResponse
import com.quickpitik.dto.photographer.PhotographerBlurryPhotoDto
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.OwnCaseRow
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.service.events.postgresTimestamp
import com.quickpitik.service.storage.StorageService
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.util.UUID

// RV-3.3 (G8): a photographer hides the blurry photos of their own batch, and undoes it. The hides are PHOTOGRAPHER holds
// on the batch: an admin's or the automatic restore never lifts them, and undo lifts only them, by the one restore rule
// (another hold or a hidden photo flag still keeps a photo out). Only a case of the caller's own batch applied in
// auto-hide, with the mode auto-hide now; anything else does nothing.
@Service
class RelevancePhotographerHideService(
    private val cases: RelevanceCaseRepository,
    private val holds: RelevanceHoldRepository,
    private val photos: PhotoRepository,
    private val switch: RelevanceSwitch,
    private val props: RelevanceProperties,
    private val storage: StorageService,
    private val storageProperties: StorageProperties,
    private val clock: Clock,
    private val transactions: TransactionTemplate,
) {

    // No transaction: each read is its own, and the preview links are signed after them.
    fun blurry(photographerId: UUID, caseId: UUID, page: PaginationParams): PaginatedResponse<PhotographerBlurryPhotoDto> {
        if (switch.mode != RelevanceMode.AUTO_HIDE) throw notFound()
        val row = cases.ownCase(caseId, photographerId)?.takeIf { it.appliedMode == AUTO } ?: throw notFound()
        val cutoff = props.junk.blurCutoff ?: return PaginatedResponse.empty(page)
        val result = holds.pageBlurryLive(row.batchId, cutoff, OffsetLimitPageable(page))
        return PaginatedResponse.of(result.content.map {
            PhotographerBlurryPhotoDto(it.photoId,
                (it.watermarkKey ?: it.thumbnailKey)?.let { key -> storage.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail) })
        }, result.totalElements, page)
    }

    // At most a page of ids, so one short transaction: hold rows first, then the photos locked in id order and hidden.
    fun hide(photographerId: UUID, caseId: UUID, photoIds: List<UUID>): PhotoActionResponse = act(photographerId, caseId, photoIds) { row, ids ->
        holds.holdByPhotographer(row.batchId, row.caseId, ids, clock.postgresTimestamp())
        photos.lockIdsForUpdate(ids)
        holds.hidePhotos(ids)
    }

    fun unhide(photographerId: UUID, caseId: UUID, photoIds: List<UUID>): PhotoActionResponse = act(photographerId, caseId, photoIds) { row, ids ->
        val released = holds.restoreByPhotographer(row.batchId, ids, clock.postgresTimestamp())
        if (released.isEmpty()) 0 else {
            photos.lockIdsForUpdate(released)
            holds.unhideReleased(released)
        }
    }

    // affected = photos whose gallery status changed.
    private fun act(photographerId: UUID, caseId: UUID, photoIds: List<UUID>, change: (OwnCaseRow, List<UUID>) -> Int): PhotoActionResponse {
        if (switch.mode != RelevanceMode.AUTO_HIDE) throw actionsOff()
        val ids = photoIds.distinct()
        if (ids.isEmpty() || ids.size > PaginationParams.MAX_LIMIT) throw badIds()
        return PhotoActionResponse(transactions.execute {
            val row = cases.ownCase(caseId, photographerId) ?: throw notFound()
            if (row.appliedMode != AUTO) throw actionsOff()
            if (holds.countInBatch(ids, row.eventId, photographerId) != ids.size.toLong()) throw badIds()
            change(row, ids)
        }!!)
    }

    private fun notFound() = NotFoundException("Case not found", ErrorCodes.NOT_FOUND)

    private fun actionsOff() = ApiException(HttpStatus.CONFLICT, ErrorCodes.RELEVANCE_ACTIONS_OFF,
        "Hiding photos needs a case applied in the relevance mode auto-hide")

    private fun badIds() = ValidationException("Send 1 to ${PaginationParams.MAX_LIMIT} photo ids, all of your photos in this event",
        field = "photoIds")

    private companion object {
        const val AUTO = "AUTO_HIDE"
    }
}
