package com.quickpitik.service.photos

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.dto.photos.toDto
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

@Service
@Transactional(readOnly = true)
class PhotoService(
    private val photoRepository: PhotoRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
) {
    fun listForEvent(
        eventId: UUID,
        bib: String?,
        pagination: PaginationParams,
        requesterUserId: UUID? = null,
    ): PaginatedResponse<PhotoDto> {
        val cleanedBib = normalizeBib(bib)
        val page = photoRepository.searchForEvent(
            eventId = eventId,
            status = PhotoStatus.LIVE,
            bib = cleanedBib,
            pageable = OffsetLimitPageable(pagination),
        )
        return toPaginatedDto(page.content, page.totalElements, pagination, requesterUserId)
    }

    fun findByEventAndPersonIds(
        eventId: UUID,
        aiPersonIds: Collection<String>,
        pagination: PaginationParams,
        requesterUserId: UUID? = null,
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
        return toPaginatedDto(page.content, page.totalElements, pagination, requesterUserId)
    }

    private fun toPaginatedDto(
        photos: List<Photo>,
        total: Long,
        pagination: PaginationParams,
        requesterUserId: UUID?,
    ): PaginatedResponse<PhotoDto> {
        val ownedIds = resolveOwnedIds(requesterUserId, photos)
        return PaginatedResponse(
            items = photos.map {
                it.toDto(
                    thumbnailUrlResolver = ::resolveThumbnailUrl,
                    cleanUrlResolver = { photo ->
                        if (photo.id in ownedIds) resolveCleanUrl(photo) else null
                    },
                )
            },
            total = total,
            offset = pagination.offset,
            limit = pagination.limit,
        )
    }

    private fun resolveOwnedIds(requesterUserId: UUID?, photos: List<Photo>): Set<UUID> {
        if (requesterUserId == null || photos.isEmpty()) return emptySet()
        return downloadGrantRepository
            .findOwnedPhotoIdsByUserAndPhotoIds(requesterUserId, photos.map { it.id })
            .toSet()
    }

    private fun resolveThumbnailUrl(photo: Photo): String? {
        val key = photo.thumbnailS3Key ?: photo.watermarkS3Key ?: photo.s3Key
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
    }

    // The clean original. Reuses the runner-download TTL since the lightbox
    // surface and the actual download share the same security envelope —
    // both are gated on proof of ownership upstream.
    private fun resolveCleanUrl(photo: Photo): String =
        storageService.presignedGetUrl(photo.s3Key, storageProperties.presignedTtl.runnerDownload)

    private fun normalizeBib(raw: String?): String {
        if (raw.isNullOrBlank()) return ""
        return raw.removePrefix("B-").removePrefix("b-").trim().uppercase()
    }
}
