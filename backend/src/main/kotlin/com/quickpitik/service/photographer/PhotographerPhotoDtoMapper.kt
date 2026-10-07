package com.quickpitik.service.photographer

import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.PhotographerLibraryPhotoDto
import com.quickpitik.dto.photographer.toLibraryDto
import com.quickpitik.entity.Photo
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Component

@Component
class PhotographerPhotoDtoMapper(
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val photoRepository: PhotoRepository,
) {
    fun toLibraryDto(photo: Photo, salesCount: Long): PhotographerLibraryPhotoDto =
        photo.toLibraryDto(salesCount, ::thumbnailUrl)

    fun thumbnailUrl(photo: Photo): String? {
        if (photo.deletionStartedAt != null) return null
        val key = photo.thumbnailS3Key ?: photo.watermarkS3Key ?: photo.s3Key
        if (!photoRepository.isDeliveryAvailable(photo.id)) return null
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
    }
}
