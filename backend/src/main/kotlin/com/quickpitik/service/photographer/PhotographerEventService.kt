package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.PhotographerDownloadDto
import com.quickpitik.dto.photographer.PhotographerEventDetailDto
import com.quickpitik.dto.photographer.PhotographerEventSummaryDto
import com.quickpitik.dto.photographer.PhotographerLibraryPhotoDto
import com.quickpitik.dto.photographer.detailDto
import com.quickpitik.dto.photographer.summaryDto
import com.quickpitik.dto.photographer.toLibraryDto
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.springframework.data.domain.Sort
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional(readOnly = true)
class PhotographerEventService(
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val eventRepository: EventRepository,
    private val photoRepository: PhotoRepository,
    private val orderItemRepository: OrderItemRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
) {
    fun listEvents(
        photographerId: UUID,
        withUploads: Boolean,
        pagination: PaginationParams,
    ): PaginatedResponse<PhotographerEventSummaryDto> {
        val page = eventPhotographerRepository.searchForPhotographer(
            photographerId = photographerId,
            withUploadsOnly = withUploads,
            pageable = OffsetLimitPageable(pagination),
        )
        if (page.isEmpty) return PaginatedResponse.empty(pagination)
        val eventIds = page.content.map { it.id.eventId }
        val eventsById = eventRepository.findAllById(eventIds).associateBy { it.id }
        val items = page.content.mapNotNull { ep ->
            val event = eventsById[ep.id.eventId] ?: return@mapNotNull null
            if (event.deletedAt != null) return@mapNotNull null
            summaryDto(event, ep)
        }
        return PaginatedResponse(
            items = items,
            total = page.totalElements,
            offset = pagination.offset,
            limit = pagination.limit,
        )
    }

    fun getEventDetail(photographerId: UUID, eventId: UUID): PhotographerEventDetailDto {
        val event = eventRepository.findById(eventId).orElse(null)?.takeIf { it.deletedAt == null }
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        val ep = loadOrEmpty(eventId, photographerId)
        // Hide events the photographer has never touched — anti-leak per Q-014.
        if (ep.photoCount == 0 && ep.firstUploadAt == null) {
            throw NotFoundException(
                code = ErrorCodes.EVENT_NOT_FOUND,
                message = "No coverage for this event",
            )
        }
        return detailDto(event, ep)
    }

    fun listPhotos(
        photographerId: UUID,
        eventId: UUID,
        order: String?,
        pagination: PaginationParams,
    ): PaginatedResponse<PhotographerLibraryPhotoDto> {
        eventRepository.findById(eventId).orElse(null)?.takeIf { it.deletedAt == null }
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        val sort = if (order.equals("oldest", ignoreCase = true)) {
            Sort.by(Sort.Direction.ASC, "uploadedAt").and(Sort.by(Sort.Direction.ASC, "id"))
        } else {
            Sort.by(Sort.Direction.DESC, "uploadedAt").and(Sort.by(Sort.Direction.ASC, "id"))
        }
        val page = photoRepository.findPhotographerLibrary(
            eventId = eventId,
            photographerId = photographerId,
            pageable = OffsetLimitPageable(pagination, sort),
        )
        if (page.isEmpty) return PaginatedResponse.empty(pagination)
        val photoIds = page.content.map { it.id }
        val salesByPhoto: Map<UUID, Long> = if (photoIds.isEmpty()) {
            emptyMap()
        } else {
            orderItemRepository.countSalesByPhotoIds(photoIds).associate { it.photoId to it.salesCount }
        }
        return PaginatedResponse(
            items = page.content.map { it.toLibraryDto(salesByPhoto[it.id] ?: 0L) },
            total = page.totalElements,
            offset = pagination.offset,
            limit = pagination.limit,
        )
    }

    fun getDownload(photographerId: UUID, photoId: UUID): PhotographerDownloadDto {
        // findFirstByIdAndPhotographerId scopes by ownership in one query —
        // a foreign photographer's photoId returns null → 404 (Q-015 anti-IDOR
        // — never leak existence by 403 vs 404).
        val photo = photoRepository.findFirstByIdAndPhotographerId(photoId, photographerId)
            ?: throw NotFoundException(code = ErrorCodes.PHOTO_NOT_FOUND, message = "Photo not found")
        val ttl = storageProperties.presignedTtl.photographerDownload
        val url = storageService.presignedGetUrl(photo.s3Key, ttl)
        return PhotographerDownloadDto(
            url = url,
            expiresAt = OffsetDateTime.now().plus(ttl),
        )
    }

    private fun loadOrEmpty(eventId: UUID, photographerId: UUID): EventPhotographer =
        eventPhotographerRepository.findById(
            com.quickpitik.entity.EventPhotographerId(eventId, photographerId),
        ).orElseGet {
            EventPhotographer(
                id = com.quickpitik.entity.EventPhotographerId(eventId, photographerId),
            )
        }
}
