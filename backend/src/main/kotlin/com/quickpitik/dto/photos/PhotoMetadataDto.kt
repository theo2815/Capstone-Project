package com.quickpitik.dto.photos

import com.quickpitik.entity.PhotoMetadata
import com.quickpitik.entity.PhotoMetadataState
import java.time.LocalDateTime
import java.util.UUID

data class PhotoMetadataFields(
    val cameraMake: String? = null,
    val cameraModel: String? = null,
    val lens: String? = null,
    val aperture: Double? = null,
    val exposureTime: String? = null,
    val iso: Long? = null,
    val author: String? = null,
    val copyright: String? = null,
    val takenAtLocal: LocalDateTime? = null,
    val takenAtOffset: String? = null,
    val flash: String? = null,
    val exposureMode: String? = null,
    val whiteBalance: String? = null,
) {
    init {
        require(aperture == null || (aperture.isFinite() && aperture > 0))
        require(iso == null || iso > 0)
    }
}

data class PhotoMetadataDto(
    val photoId: UUID,
    val status: String,
    val originalFilename: String?,
    val fileSizeBytes: Long?,
    val fields: PhotoMetadataFields?,
)

fun PhotoMetadata?.toDto(photoId: UUID = requireNotNull(this).photoId): PhotoMetadataDto {
    val status = when (this?.state) {
        null, PhotoMetadataState.UNAVAILABLE -> "unavailable"
        PhotoMetadataState.PROCESSING, PhotoMetadataState.PENDING -> "pending"
        else -> state.name.lowercase()
    }
    return PhotoMetadataDto(
        photoId = photoId,
        status = status,
        originalFilename = this?.originalFilename,
        fileSizeBytes = this?.originalSizeBytes,
        fields = when (this?.state) {
            PhotoMetadataState.READY -> fields ?: PhotoMetadataFields()
            PhotoMetadataState.EMPTY -> PhotoMetadataFields()
            else -> null
        },
    )
}
