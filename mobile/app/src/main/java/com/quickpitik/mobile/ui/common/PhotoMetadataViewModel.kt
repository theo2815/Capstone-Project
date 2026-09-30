package com.quickpitik.mobile.ui.common

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.LocalPhotoMetadata
import com.quickpitik.mobile.data.local.LocalPhotoMetadataReader
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.data.remote.PhotoMetadataDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.repository.PhotoMetadataRepository
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.launch
import retrofit2.HttpException

sealed interface PhotoMetadataState {
    data object Closed : PhotoMetadataState
    data class Loading(val photoId: String, val access: String) : PhotoMetadataState
    data class LocalLoading(val uploadRecordId: Long) : PhotoMetadataState
    data class Data(val metadata: PhotoMetadataDto) : PhotoMetadataState
    data class Pending(val metadata: PhotoMetadataDto) : PhotoMetadataState
    data class Empty(val metadata: PhotoMetadataDto) : PhotoMetadataState
    data class Failed(val message: String) : PhotoMetadataState
    data class Unavailable(val metadata: PhotoMetadataDto) : PhotoMetadataState
    data object Denied : PhotoMetadataState
    data class LocalData(val metadata: LocalPhotoMetadata) : PhotoMetadataState
    data class LocalEmpty(val metadata: LocalPhotoMetadata) : PhotoMetadataState
    data class LocalUnavailable(val metadata: LocalPhotoMetadata) : PhotoMetadataState
}

class PhotoMetadataViewModel @JvmOverloads constructor(
    application: Application,
    private val getRecord: suspend (Long) -> UploadRecord? =
        { AppDatabase.getDatabase(application).uploadQueueDao().getRecordById(it) },
    private val readLocal: suspend (UploadRecord) -> LocalPhotoMetadata =
        LocalPhotoMetadataReader(application)::read,
    private val load: suspend (String, Boolean) -> PhotoMetadataDto =
        PhotoMetadataRepository(SessionManager.getInstance(application))::getPhotoMetadata,
) : AndroidViewModel(application) {
    private val session = SessionManager.getInstance(application)
    private val _state = MutableStateFlow<PhotoMetadataState>(PhotoMetadataState.Closed)
    val state: StateFlow<PhotoMetadataState> = _state
    @Volatile private var selected: Pair<String, String>? = null
    @Volatile private var selectedLocal: Long? = null
    private var job: Job? = null
    @Volatile private var request = 0

    init {
        viewModelScope.launch(start = CoroutineStart.UNDISPATCHED) {
            combine(session.authChanges, RetrofitClient.backendChanges) { auth, backend -> auth to backend }
                .drop(1).collect { clear() }
        }
    }

    fun selectRemote(photoId: String, access: String) {
        clear()
        if (photoId.isBlank() || access !in setOf("owner", "public")) return
        selected = photoId to access
        fetch()
    }

    fun selectLocal(uploadRecordId: Long) {
        clear()
        if (uploadRecordId <= 0) return
        selectedLocal = uploadRecordId
        fetch()
    }

    fun refresh() {
        if (selected != null || selectedLocal != null) fetch()
    }

    @Synchronized fun clear() {
        request++
        job?.cancel()
        selected = null
        selectedLocal = null
        _state.value = PhotoMetadataState.Closed
    }

    private fun fetch() {
        val remote = selected
        val localId = selectedLocal
        if (remote == null && localId == null) return
        val photoId = remote?.first
        val access = remote?.second
        val token = session.getAccessToken()
        val auth = token?.let(session::authSnapshot)
        val backend = RetrofitClient.BASE_URL
        val currentRequest = ++request
        job?.cancel()
        _state.value = if (localId != null) PhotoMetadataState.LocalLoading(localId)
            else PhotoMetadataState.Loading(photoId!!, access!!)
        fun current() = currentRequest == request && selected == remote && selectedLocal == localId &&
            backend == RetrofitClient.BASE_URL &&
            (if (token == null) session.getAccessToken() == null else auth != null && session.isCurrent(auth))
        fun publish(next: PhotoMetadataState) = synchronized(this) {
            if (current()) _state.value = next
            else if (currentRequest == request) _state.value = PhotoMetadataState.Closed
        }
        job = viewModelScope.launch(Dispatchers.IO) {
            try {
                if (localId != null) {
                    val row = getRecord(localId)
                    if (row == null || auth?.owner != row.photographerId ||
                        session.getUserRole() != "PHOTOGRAPHER") {
                        publish(PhotoMetadataState.LocalUnavailable(LocalPhotoMetadata("unavailable", null, null, null)))
                        return@launch
                    }
                    val local = readLocal(row)
                    if (local.status != "unavailable") {
                        publish(when (local.status) {
                            "ready" -> PhotoMetadataState.LocalData(local)
                            "empty" -> PhotoMetadataState.LocalEmpty(local)
                            else -> PhotoMetadataState.Failed("Could not prepare metadata. Try again.")
                        })
                        return@launch
                    }
                    val receipt = row.uploadedPhotoId?.takeIf { row.uploadStatus == "COMPLETED" &&
                        row.uploadedBackend == backend && it.isNotBlank() }
                    if (receipt == null) {
                        publish(PhotoMetadataState.LocalUnavailable(local))
                        return@launch
                    }
                    if (!current()) return@launch
                    val result = load(receipt, true)
                    publish(remoteState(result))
                    return@launch
                }
                val result = load(photoId!!, access == "owner")
                publish(remoteState(result))
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                publish(if (access == "public" && e is HttpException && e.code() == 404)
                    PhotoMetadataState.Denied
                else PhotoMetadataState.Failed("Could not load metadata. Try again."))
            }
        }
    }

    private fun remoteState(result: PhotoMetadataDto): PhotoMetadataState =
        when (result.status.lowercase()) {
            "ready" -> PhotoMetadataState.Data(result)
            "pending" -> PhotoMetadataState.Pending(result)
            "empty" -> PhotoMetadataState.Empty(result)
            "failed" -> PhotoMetadataState.Failed("Could not prepare metadata. Try again.")
            else -> PhotoMetadataState.Unavailable(result)
        }
}
