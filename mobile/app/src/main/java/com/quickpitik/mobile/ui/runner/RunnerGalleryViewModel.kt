package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.RetrofitClient
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

    init {
        fetchPublicEvents()
    }

    fun fetchPublicEvents() {
        viewModelScope.launch {
            _eventsState.value = RunnerEventsState.Loading
            try {
                // Fetch public ACTIVE status events
                val response = RetrofitClient.apiService.getPublicEvents("ACTIVE")
                if (response.success && response.data != null) {
                    val list = response.data.items
                    _eventsState.value = RunnerEventsState.Success(list)
                    if (_activeEvent.value == null && list.isNotEmpty()) {
                        _activeEvent.value = list.first()
                        // Initial photo stream load
                        searchByBib("")
                    }
                } else {
                    _eventsState.value = RunnerEventsState.Error(response.error ?: "Failed to load active events.")
                }
            } catch (e: Exception) {
                _eventsState.value = RunnerEventsState.Error(e.localizedMessage ?: "Failed to connect to backend server.")
            }
        }
    }

    fun selectEvent(event: EventDto) {
        _activeEvent.value = event
        searchByBib("")
    }

    fun searchByBib(bib: String) {
        val event = _activeEvent.value ?: return
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
                _searchState.value = PhotosSearchState.Error(e.localizedMessage ?: "Failed to query event photos.")
            }
        }
    }

    fun searchBySelfie(selfieFile: File) {
        val event = _activeEvent.value ?: return
        viewModelScope.launch {
            _searchState.value = PhotosSearchState.Loading
            try {
                val requestFile = selfieFile.asRequestBody("image/jpeg".toMediaTypeOrNull())
                val selfiePart = MultipartBody.Part.createFormData("selfie", selfieFile.name, requestFile)

                val response = RetrofitClient.apiService.searchPhotosByFace(
                    slug = event.slug,
                    selfie = selfiePart
                )
                if (response.success && response.data != null) {
                    _searchState.value = PhotosSearchState.Success(response.data.items)
                } else {
                    _searchState.value = PhotosSearchState.Error(response.error ?: "AI Face Recognition returned error.")
                }
            } catch (e: Exception) {
                _searchState.value = PhotosSearchState.Error(e.localizedMessage ?: "AI Service connection timed out.")
            }
        }
    }

    fun simulateSelfieSearch() {
        _activeEvent.value ?: return
        viewModelScope.launch {
            try {
                // 1. Create a simulated physical JPEG image on phone cache storage
                val cacheDir = getApplication<Application>().cacheDir
                val mockFile = File(cacheDir, "simulated_selfie_${System.currentTimeMillis()}.jpg")
                
                // Write standard JPEG header bytes
                val jpegHeader = byteArrayOf(
                    0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0xE0.toByte(), 
                    0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00
                )
                mockFile.writeBytes(jpegHeader)

                // 2. Fire the multipart search
                searchBySelfie(mockFile)
            } catch (e: Exception) {
                _searchState.value = PhotosSearchState.Error("Simulation setup failed: ${e.localizedMessage}")
            }
        }
    }
}
