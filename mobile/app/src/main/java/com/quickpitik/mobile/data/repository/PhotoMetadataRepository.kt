package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.EventMetadataVisibilityDto
import com.quickpitik.mobile.data.remote.MetadataSettingsDto
import com.quickpitik.mobile.data.remote.PhotoMetadataDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.toResult

class PhotoMetadataRepository(private val session: SessionManager) {
    private val api get() = RetrofitClient.apiService

    private suspend fun <T> owned(block: suspend (String) -> T): T {
        val token = session.getAccessToken() ?: error("Photographer session required")
        check(session.getUserRole() == "PHOTOGRAPHER") { "Photographer session required" }
        val snapshot = session.authSnapshot(token) ?: error("Photographer session required")
        val backend = RetrofitClient.BASE_URL
        val result = block("Bearer $token")
        check(session.isCurrent(snapshot) && backend == RetrofitClient.BASE_URL) { "Session changed" }
        return result
    }

    suspend fun getMetadataSettings(): MetadataSettingsDto =
        owned { api.getMetadataSettings(it).toResult("Could not load metadata settings").getOrThrow() }

    suspend fun setMetadataSettings(value: Boolean): MetadataSettingsDto =
        owned { api.setMetadataSettings(it, MetadataSettingsDto(value)).toResult("Could not save metadata settings").getOrThrow() }

    suspend fun getEventVisibility(eventId: String): EventMetadataVisibilityDto =
        owned { api.getEventMetadataVisibility(it, eventId).toResult("Could not load metadata visibility").getOrThrow() }

    suspend fun setEventVisibility(eventId: String, value: Boolean): EventMetadataVisibilityDto =
        owned { api.setEventMetadataVisibility(it, eventId, EventMetadataVisibilityDto(value)).toResult("Could not save metadata visibility").getOrThrow() }

    suspend fun getPhotoMetadata(photoId: String, owner: Boolean): PhotoMetadataDto =
        if (owner) owned { api.getOwnedPhotoMetadata(it, photoId).toResult("Could not load photo metadata").getOrThrow() }
        else api.getPublicPhotoMetadata(photoId).toResult("Could not load photo metadata").getOrThrow()
}
