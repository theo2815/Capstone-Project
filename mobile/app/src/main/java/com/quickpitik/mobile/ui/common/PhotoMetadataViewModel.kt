package com.quickpitik.mobile.ui.common

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.PhotoMetadataDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.repository.PhotoMetadataRepository
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import retrofit2.HttpException

sealed interface PhotoMetadataState {
    data object Closed : PhotoMetadataState
    data class Loading(val photoId: String, val access: String) : PhotoMetadataState
    data class Data(val metadata: PhotoMetadataDto) : PhotoMetadataState
    data class Pending(val metadata: PhotoMetadataDto) : PhotoMetadataState
    data class Empty(val metadata: PhotoMetadataDto) : PhotoMetadataState
    data class Failed(val message: String) : PhotoMetadataState
    data object Unavailable : PhotoMetadataState
    data object Denied : PhotoMetadataState
}

class PhotoMetadataViewModel(application: Application) : AndroidViewModel(application) {
    private val session = SessionManager.getInstance(application)
    private val repository = PhotoMetadataRepository(session)
    private val _state = MutableStateFlow<PhotoMetadataState>(PhotoMetadataState.Closed)
    val state: StateFlow<PhotoMetadataState> = _state
    @Volatile private var selected: Pair<String, String>? = null
    private var job: Job? = null
    @Volatile private var request = 0

    fun selectRemote(photoId: String, access: String) {
        clear()
        if (photoId.isBlank() || access !in setOf("owner", "public")) return
        selected = photoId to access
        fetch()
    }

    fun refresh() {
        if (selected != null) fetch()
    }

    @Synchronized fun clear() {
        request++
        job?.cancel()
        selected = null
        _state.value = PhotoMetadataState.Closed
    }

    private fun fetch() {
        val (photoId, access) = selected ?: return
        val token = session.getAccessToken()
        val auth = token?.let(session::authSnapshot)
        val backend = RetrofitClient.BASE_URL
        val currentRequest = ++request
        job?.cancel()
        _state.value = PhotoMetadataState.Loading(photoId, access)
        fun current() = currentRequest == request && selected == (photoId to access) &&
            backend == RetrofitClient.BASE_URL &&
            (if (token == null) session.getAccessToken() == null else auth != null && session.isCurrent(auth))
        fun publish(next: PhotoMetadataState) = synchronized(this) {
            if (current()) _state.value = next
            else if (currentRequest == request) _state.value = PhotoMetadataState.Closed
        }
        job = viewModelScope.launch(Dispatchers.IO) {
            try {
                val result = repository.getPhotoMetadata(photoId, owner = access == "owner")
                publish(when (result.status.lowercase()) {
                    "ready" -> PhotoMetadataState.Data(result)
                    "pending" -> PhotoMetadataState.Pending(result)
                    "empty" -> PhotoMetadataState.Empty(result)
                    "failed" -> PhotoMetadataState.Failed("Could not prepare metadata. Try again.")
                    else -> PhotoMetadataState.Unavailable
                })
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                publish(if (access == "public" && e is HttpException && e.code() == 404)
                    PhotoMetadataState.Denied
                else PhotoMetadataState.Failed("Could not load metadata. Try again."))
            }
        }
    }
}
