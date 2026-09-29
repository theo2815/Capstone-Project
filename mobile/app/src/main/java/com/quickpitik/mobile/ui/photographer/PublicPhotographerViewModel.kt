package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.ApiResponseEnvelope
import com.quickpitik.mobile.data.remote.EventDetailDto
import com.quickpitik.mobile.data.remote.PaginatedResponse
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.PhotographerProfileDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.apiCall
import com.quickpitik.mobile.data.remote.uiMessage
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.drop
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
 * The profile is anonymous; gallery photos carry optional session auth so the
 * backend can return the viewer's metadata access hint.
 */
class PublicPhotographerViewModel @JvmOverloads constructor(
    application: Application,
    private val loadPhotos: suspend (String, String, String?) -> ApiResponseEnvelope<PaginatedResponse<PhotoDto>> =
        { handle, slug, bearer -> RetrofitClient.apiService.getPublicPhotographerEventPhotos(handle, slug, token = bearer) },
) : AndroidViewModel(application) {
    private val session = SessionManager.getInstance(application)
    private val _publicProfileState = MutableStateFlow<PublicProfileState>(PublicProfileState.Loading)
    val publicProfileState: StateFlow<PublicProfileState> = _publicProfileState

    private val _profileEventPhotosState = MutableStateFlow<ProfileEventPhotosState>(ProfileEventPhotosState.Loading)
    val profileEventPhotosState: StateFlow<ProfileEventPhotosState> = _profileEventPhotosState
    private var gallery: Pair<String, String>? = null
    private var galleryJob: Job? = null
    private var galleryRequest = 0L

    init {
        viewModelScope.launch(start = CoroutineStart.UNDISPATCHED) {
            combine(session.authChanges, RetrofitClient.backendChanges) { auth, backend -> auth to backend }
                .drop(1).collect {
                    gallery?.let { (handle, slug) -> fetchProfileEventPhotos(handle, slug) }
                }
        }
    }

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

    @Synchronized fun fetchProfileEventPhotos(handle: String, slug: String) {
        gallery = handle to slug
        clearGalleryResult()
        val request = galleryRequest
        val backend = RetrofitClient.BASE_URL
        val generation = session.authChanges.value
        val token = session.getAccessToken()
        val bearer = token?.let { "Bearer $it" }
        galleryJob = viewModelScope.launch(Dispatchers.IO) {
            val result = apiCall("Failed to load photos.") {
                loadPhotos(handle, slug, bearer)
            }
            synchronized(this@PublicPhotographerViewModel) {
                if (request != galleryRequest || gallery != (handle to slug) ||
                    backend != RetrofitClient.BASE_URL || generation != session.authChanges.value
                ) return@synchronized
                result.fold(
                    onSuccess = { _profileEventPhotosState.value = ProfileEventPhotosState.Success(it.items) },
                    onFailure = { _profileEventPhotosState.value = ProfileEventPhotosState.Error(it.uiMessage) },
                )
            }
        }
    }

    @Synchronized fun clearGallery() {
        gallery = null
        clearGalleryResult()
    }

    @Synchronized private fun clearGalleryResult() {
        galleryRequest++
        galleryJob?.cancel()
        _profileEventPhotosState.value = ProfileEventPhotosState.Loading
    }
}
