package com.quickpitik.mobile.data.remote

data class MetadataSettingsDto(val metadataPublicByDefault: Boolean)
data class EventMetadataVisibilityDto(val metadataPublic: Boolean)

data class PhotoMetadataFieldsDto(
    val cameraMake: String?,
    val cameraModel: String?,
    val lens: String?,
    val aperture: Double?,
    val exposureTime: String?,
    val iso: Long?,
    val author: String?,
    val copyright: String?,
    val takenAtLocal: String?,
    val takenAtOffset: String?,
    val flash: String?,
    val exposureMode: String?,
    val whiteBalance: String?,
)

data class PhotoMetadataDto(
    val photoId: String,
    val status: String,
    val originalFilename: String?,
    val fileSizeBytes: Long?,
    val fields: PhotoMetadataFieldsDto?,
)
