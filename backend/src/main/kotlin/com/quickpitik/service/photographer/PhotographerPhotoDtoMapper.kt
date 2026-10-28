package com.quickpitik.service.photographer

import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.PhotographerLibraryPhotoDto
import com.quickpitik.dto.photographer.toLibraryDto
import com.quickpitik.entity.Photo
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Component

@Component
class PhotographerPhotoDtoMapper(
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
) {
    fun toLibraryDto(photo: Photo, salesCount: Long, mediaAvailable: Boolean): PhotographerLibraryPhotoDto =
        photo.toLibraryDto(salesCount, mediaAvailable) { thumbnailUrl(it, mediaAvailable) }

    fun thumbnailUrl(photo: Photo, mediaAvailable: Boolean): String? {
        if (!mediaAvailable) return null
        val key = photo.watermarkS3Key ?: photo.thumbnailS3Key ?: photo.s3Key
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
    }
}
