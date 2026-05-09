package com.quickpitik.service.photos

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.dto.photos.toDto
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

@Service
@Transactional(readOnly = true)
class PhotoService(
    private val photoRepository: PhotoRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
) {
    fun listForEvent(
        eventId: UUID,
        bib: String?,
        pagination: PaginationParams,
    ): PaginatedResponse<PhotoDto> {
        val cleanedBib = normalizeBib(bib)
        val page = photoRepository.searchForEvent(
            eventId = eventId,
            status = PhotoStatus.LIVE,
            bib = cleanedBib,
            pageable = OffsetLimitPageable(pagination),
        )
        return PaginatedResponse(
            items = page.content.map { it.toDto(::resolveThumbnailUrl) },
            total = page.totalElements,
            offset = pagination.offset,
            limit = pagination.limit,
        )
    }

    fun findByEventAndPersonIds(
        eventId: UUID,
        aiPersonIds: Collection<String>,
        pagination: PaginationParams,
    ): PaginatedResponse<PhotoDto> {
        if (aiPersonIds.isEmpty()) {
            return PaginatedResponse.empty(pagination)
        }
        val page = photoRepository.findByEventAndPersonIds(
            eventId = eventId,
            status = PhotoStatus.LIVE,
            aiPersonIds = aiPersonIds,
            pageable = OffsetLimitPageable(pagination),
        )
        return PaginatedResponse(
            items = page.content.map { it.toDto(::resolveThumbnailUrl) },
            total = page.totalElements,
            offset = pagination.offset,
            limit = pagination.limit,
        )
    }

    private fun resolveThumbnailUrl(photo: Photo): String? {
        val key = photo.thumbnailS3Key ?: photo.watermarkS3Key ?: photo.s3Key
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
    }

    private fun normalizeBib(raw: String?): String {
        if (raw.isNullOrBlank()) return ""
        return raw.removePrefix("B-").removePrefix("b-").trim().uppercase()
    }
}
