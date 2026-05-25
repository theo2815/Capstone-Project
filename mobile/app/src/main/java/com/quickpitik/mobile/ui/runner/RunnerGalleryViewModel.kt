package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.SelfieRefDto
import com.quickpitik.mobile.data.remote.SearchByFaceJsonRequest
import com.quickpitik.mobile.data.local.SessionManager
import android.net.Uri
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File

sealed class RunnerEventsState {
    object Loading : RunnerEventsState()
    data class Success(val events: List<EventDto>) : RunnerEventsState()
    data class Error(val message: String) : RunnerEventsState()
}

sealed class PhotosSearchState {
    object Idle : PhotosSearchState()
    object Loading : PhotosSearchState()
    data class Success(val photos: List<PhotoDto>) : PhotosSearchState()
    data class Error(val message: String) : PhotosSearchState()
}

class RunnerGalleryViewModel(application: Application) : AndroidViewModel(application) {
    private val _eventsState = MutableStateFlow<RunnerEventsState>(RunnerEventsState.Loading)
    val eventsState: StateFlow<RunnerEventsState> = _eventsState

    private val _activeEvent = MutableStateFlow<EventDto?>(null)
    val activeEvent: StateFlow<EventDto?> = _activeEvent

    private val _searchState = MutableStateFlow<PhotosSearchState>(PhotosSearchState.Idle)
    val searchState: StateFlow<PhotosSearchState> = _searchState

    private val _isFiltered = MutableStateFlow(false)
    val isFiltered: StateFlow<Boolean> = _isFiltered

    init {
        fetchPublicEvents()
    }

    fun fetchPublicEvents() {
        viewModelScope.launch {
            _eventsState.value = RunnerEventsState.Loading
            try {
                // Browse-first, like the website's /events: pull every publicly
                // visible lifecycle bucket so the discovery screen can segment
                // Upcoming / Live / Recent / Archive. Selection is explicit (a
                // card tap calls selectEvent) — we no longer auto-select the first
                // event, since the runner now lands on the browse screen, not the
                // cockpit. Backend EventController.parseStatusList accepts the CSV.
                val response = RetrofitClient.apiService.getPublicEvents(
                    status = "ACTIVE,COMPLETED,ARCHIVED",
                    offset = 0,
                    limit = 200
                )
                if (response.success && response.data != null) {
                    _eventsState.value = RunnerEventsState.Success(response.data.items)
                } else {
                    _eventsState.value = RunnerEventsState.Error(response.error ?: "Failed to load events.")
                }
            } catch (e: Exception) {
                _eventsState.value = RunnerEventsState.Error(e.localizedMessage ?: "Failed to connect to backend server.")
            }
        }
    }

    // Resolve a loaded event by slug. Used by the profile race log's "Open →" to
    // jump straight into the cockpit for a saved/purchased event without a second
    // round-trip. Returns null if events haven't loaded yet or there's no match.
    fun eventBySlug(slug: String): EventDto? =
        (_eventsState.value as? RunnerEventsState.Success)?.events?.firstOrNull { it.slug == slug }

    fun selectEvent(event: EventDto) {
        _activeEvent.value = event
        _isFiltered.value = false
        searchByBib("")
    }

    fun clearSelectedEvent() {
        _activeEvent.value = null
        _searchState.value = PhotosSearchState.Idle
        _isFiltered.value = false
    }

    fun searchByBib(bib: String) {
        val event = _activeEvent.value ?: return
        _isFiltered.value = bib.trim().isNotEmpty()
        viewModelScope.launch {
            _searchState.value = PhotosSearchState.Loading
            try {
                val query = bib.trim().ifEmpty { null }
                val response = RetrofitClient.apiService.getEventPhotos(
                    slug = event.slug,
                    bib = query
                )
                if (response.success && response.data != null) {
                    _searchState.value = PhotosSearchState.Success(response.data.items)
                } else {
                    _searchState.value = PhotosSearchState.Error(response.error ?: "Search lookup failed.")
                }
            } catch (e: Exception) {
                _searchState.value = PhotosSearchState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    fun clearFilter() {
        _isFiltered.value = false
        searchByBib("")
    }

    fun searchBySelfie(selfieFile: File) {
        val event = _activeEvent.value ?: return
        _isFiltered.value = true
        val application = getApplication<Application>()
        val sessionManager = SessionManager.getInstance(application)
        val token = sessionManager.getAccessToken()
        if (token == null) {
            _searchState.value = PhotosSearchState.Error("Authentication token not found. Please log in.")
            return
        }
        viewModelScope.launch {
            _searchState.value = PhotosSearchState.Loading
            try {
                val requestFile = selfieFile.asRequestBody("image/jpeg".toMediaTypeOrNull())
                val selfiePart = MultipartBody.Part.createFormData("selfie", selfieFile.name, requestFile)

                val response = RetrofitClient.apiService.searchPhotosByFace(
                    token = "Bearer $token",
                    slug = event.slug,
                    selfie = selfiePart
                )
                if (response.success && response.data != null) {
                    _searchState.value = PhotosSearchState.Success(response.data.items)
                } else {
                    _searchState.value = PhotosSearchState.Error(response.error ?: "AI Face Recognition returned error.")
                }
            } catch (e: Exception) {
                _searchState.value = PhotosSearchState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    // Real selfie capture/upload path: read the bytes the camera (or gallery
    // picker) handed us via the content URI, spool them to a cache file, then
    // run the same multipart face-search the stored-selfie path uses.
    fun searchBySelfieUri(uri: Uri) {
        _activeEvent.value ?: return
        viewModelScope.launch {
            _searchState.value = PhotosSearchState.Loading
            try {
                val resolver = getApplication<Application>().contentResolver
                val bytes = resolver.openInputStream(uri)?.use { it.readBytes() }
                if (bytes == null || bytes.isEmpty()) {
                    _searchState.value = PhotosSearchState.Error("Couldn't read that photo. Please try again.")
                    return@launch
                }
                val cacheFile = File(
                    getApplication<Application>().cacheDir,
                    "selfie_search_${System.currentTimeMillis()}.jpg"
                )
                cacheFile.writeBytes(bytes)
                searchBySelfie(cacheFile)
            } catch (e: Exception) {
                _searchState.value = PhotosSearchState.Error(e.localizedMessage ?: "Failed to process the selfie image.")
            }
        }
    }

    fun searchByStoredSelfie() {
        val event = _activeEvent.value ?: return
        _isFiltered.value = true
        val application = getApplication<Application>()
        val sessionManager = SessionManager.getInstance(application)
        val token = sessionManager.getAccessToken()
        if (token == null) {
            _searchState.value = PhotosSearchState.Error("Authentication token not found. Please log in.")
            return
        }
        viewModelScope.launch {
            _searchState.value = PhotosSearchState.Loading
            try {
                // 1. Fetch user's selfies to find the primary selfie ID
                val selfiesResponse = RetrofitClient.apiService.getSelfies("Bearer $token")
                if (!selfiesResponse.success || selfiesResponse.data == null) {
                    _searchState.value = PhotosSearchState.Error(selfiesResponse.error ?: "Failed to retrieve user selfies.")
                    return@launch
                }
                
                val primarySelfie = selfiesResponse.data.find { it.isPrimary }
                if (primarySelfie == null) {
                    _searchState.value = PhotosSearchState.Error("No primary selfie set. Please upload a selfie and set it as primary first.")
                    return@launch
                }
                
                // 2. Perform the JSON search using the primary selfie's ID
                val response = RetrofitClient.apiService.searchPhotosByFaceJson(
                    token = "Bearer $token",
                    slug = event.slug,
                    request = SearchByFaceJsonRequest(selfieId = primarySelfie.id)
                )
                if (response.success && response.data != null) {
                    _searchState.value = PhotosSearchState.Success(response.data.items)
                } else {
                    _searchState.value = PhotosSearchState.Error(response.error ?: "AI Face search returned an error.")
                }
            } catch (e: Exception) {
                _searchState.value = PhotosSearchState.Error(e.localizedMessage ?: "Failed to query event photos with stored selfie.")
            }
        }
    }
}
