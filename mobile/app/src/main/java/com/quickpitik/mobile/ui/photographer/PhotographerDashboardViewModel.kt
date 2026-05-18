package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.work.Constraints
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.worker.PhotoUploadWorker
import java.io.File
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

sealed class EventsState {
    object Loading : EventsState()
    data class Success(val events: List<PhotographerEventSummaryDto>) : EventsState()
    data class Error(val message: String) : EventsState()
}

data class QueueStats(
    val syncedCount: Int = 0,
    val queuedCount: Int = 0,
    val uploadingCount: Int = 0,
    val failedCount: Int = 0,
    val totalCount: Int = 0,
    val progress: Float = 0f
)

class PhotographerDashboardViewModel(application: Application) : AndroidViewModel(application) {
    private val database = AppDatabase.getDatabase(application)
    private val sessionManager = SessionManager.getInstance(application)
    private val workManager = WorkManager.getInstance(application)

    private val _eventsState = MutableStateFlow<EventsState>(EventsState.Loading)
    val eventsState: StateFlow<EventsState> = _eventsState

    private val _activeEvent = MutableStateFlow<PhotographerEventSummaryDto?>(null)
    val activeEvent: StateFlow<PhotographerEventSummaryDto?> = _activeEvent

    private val _queueStats = MutableStateFlow(QueueStats())
    val queueStats: StateFlow<QueueStats> = _queueStats

    init {
        fetchEvents()
        observeQueue()
    }

    fun fetchEvents() {
        viewModelScope.launch {
            _eventsState.value = EventsState.Loading
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _eventsState.value = EventsState.Error("No valid session. Please log in again.")
                return@launch
            }

            try {
                val response = RetrofitClient.apiService.getPhotographerEvents("Bearer $token")
                if (response.success && response.data != null) {
                    val list = response.data.items
                    _eventsState.value = EventsState.Success(list)
                    if (_activeEvent.value == null && list.isNotEmpty()) {
                        _activeEvent.value = list.first()
                      }
                } else {
                    _eventsState.value = EventsState.Error(response.error ?: "Failed to load events.")
                }
            } catch (e: Exception) {
                _eventsState.value = EventsState.Error(e.localizedMessage ?: "Failed to connect to server.")
            }
        }
    }

    fun selectEvent(event: PhotographerEventSummaryDto) {
        _activeEvent.value = event
    }

    private fun observeQueue() {
        viewModelScope.launch {
            database.uploadQueueDao().getAllRecords().collectLatest { records ->
                if (records.isEmpty()) {
                    _queueStats.value = QueueStats()
                    return@collectLatest
                }

                val synced = records.count { it.uploadStatus == "COMPLETED" }
                val queued = records.count { it.uploadStatus == "QUEUED" }
                val uploading = records.count { it.uploadStatus == "UPLOADING" }
                val failed = records.count { it.uploadStatus == "FAILED" }
                val total = records.size

                val progress = if (total > 0) synced.toFloat() / total.toFloat() else 0f

                _queueStats.value = QueueStats(
                    syncedCount = synced,
                    queuedCount = queued,
                    uploadingCount = uploading,
                    failedCount = failed,
                    totalCount = total,
                    progress = progress
                )
            }
        }
    }

    fun runSyncEngine() {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()

        val syncRequest = OneTimeWorkRequestBuilder<PhotoUploadWorker>()
            .setConstraints(constraints)
            .build()

        workManager.enqueue(syncRequest)
    }

    fun simulatePhotoCapture() {
        val event = _activeEvent.value ?: return
        viewModelScope.launch {
            try {
                // 1. Create a simulated physical JPEG image on phone cache storage
                val cacheDir = getApplication<Application>().cacheDir
                val mockFile = File(cacheDir, "simulated_dslr_${System.currentTimeMillis()}.jpg")
                
                // Write valid JPEG signature byte headers so S3/Spring processes the multi-part payload cleanly
                val jpegHeader = byteArrayOf(
                    0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0xE0.toByte(), 
                    0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00
                )
                mockFile.writeBytes(jpegHeader)

                // 2. Insert as a "QUEUED" record in local SQLite database
                database.uploadQueueDao().insertRecord(
                    UploadRecord(
                        filePath = mockFile.absolutePath,
                        eventId = event.id,
                        photographerId = "simulated_photographer",
                        captureTimestamp = System.currentTimeMillis(),
                        uploadStatus = "QUEUED"
                    )
                )
            } catch (e: Exception) {
                // Fail silently or log error during simulation
            }
        }
    }
}
