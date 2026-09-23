package com.quickpitik.mobile.ui.photographer

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.remote.EventDetailDto
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.PhotographerProfileDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.apiCall
import com.quickpitik.mobile.data.remote.uiMessage
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

sealed class PublicProfileState {
    object Loading : PublicProfileState()
    data class Success(val profile: PhotographerProfileDto) : PublicProfileState()
    data class Error(val message: String) : PublicProfileState()
}

sealed class ProfileEventPhotosState {
    object Loading : ProfileEventPhotosState()
    data class Success(val photos: List<PhotoDto>) : ProfileEventPhotosState()
    data class Error(val message: String) : ProfileEventPhotosState()
}

/**
 * Backs the public photographer profile at `/{handle}` and its per-event
 * galleries. Split out of [PhotographerDashboardViewModel] (2026-08-15) so a
 * RUNNER can open the screen too: the dashboard VM's `init {}` fires seven
 * photographer-scoped fetches and opens the photographer inbox socket, none of
 * which a runner session may touch.
 *
 * Both endpoints are public — no Authorization header — so this works signed
 * out as well as under either role.
 */
class PublicPhotographerViewModel : ViewModel() {
    private val _publicProfileState = MutableStateFlow<PublicProfileState>(PublicProfileState.Loading)
    val publicProfileState: StateFlow<PublicProfileState> = _publicProfileState

    private val _profileEventPhotosState = MutableStateFlow<ProfileEventPhotosState>(ProfileEventPhotosState.Loading)
    val profileEventPhotosState: StateFlow<ProfileEventPhotosState> = _profileEventPhotosState

    // The coverage row from /public/photographers/{handle} carries only the
    // SLUG — the real event name and the event id (which add-to-cart needs)
    // come from the public GET /events/{slug}. Best-effort: null just means
    // the title falls back to the prettified slug and commerce stays off.
    private val _galleryEventDetail = MutableStateFlow<EventDetailDto?>(null)
    val galleryEventDetail: StateFlow<EventDetailDto?> = _galleryEventDetail

    fun fetchGalleryEventDetail(slug: String) {
        _galleryEventDetail.value = null
        viewModelScope.launch {
            runCatching { RetrofitClient.apiService.getEventDetail(slug) }
                .getOrNull()
                ?.takeIf { it.success && it.data != null }
                ?.let { _galleryEventDetail.value = it.data }
        }
    }

    fun fetchPublicProfile(handle: String) {
        viewModelScope.launch {
            _publicProfileState.value = PublicProfileState.Loading
            apiCall("Failed to load profile.") {
                RetrofitClient.apiService.getPublicPhotographerProfile(handle)
            }.fold(
                onSuccess = { _publicProfileState.value = PublicProfileState.Success(it) },
                onFailure = { _publicProfileState.value = PublicProfileState.Error(it.uiMessage) },
            )
        }
    }

    fun fetchProfileEventPhotos(handle: String, slug: String) {
        viewModelScope.launch {
            _profileEventPhotosState.value = ProfileEventPhotosState.Loading
            apiCall("Failed to load photos.") {
                RetrofitClient.apiService.getPublicPhotographerEventPhotos(handle, slug)
            }.fold(
                onSuccess = { _profileEventPhotosState.value = ProfileEventPhotosState.Success(it.items) },
                onFailure = { _profileEventPhotosState.value = ProfileEventPhotosState.Error(it.uiMessage) },
            )
        }
    }
}
