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
import com.quickpitik.mobile.data.remote.EarningsOverviewDto
import com.quickpitik.mobile.data.remote.PayoutBalanceDto
import com.quickpitik.mobile.data.remote.PhotographerPayoutDto
import com.quickpitik.mobile.data.remote.PhotographerTransactionDto
import com.quickpitik.mobile.worker.PhotoUploadWorker
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

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

sealed class EarningsUiState {
    object Loading : EarningsUiState()
    data class Success(
        val overview: EarningsOverviewDto,
        val balance: PayoutBalanceDto,
        val transactions: List<PhotographerTransactionDto>
    ) : EarningsUiState()
    data class Error(val message: String) : EarningsUiState()
}

sealed class VerificationUiState {
    object Loading : VerificationUiState()
    data class Success(val verification: com.quickpitik.mobile.data.remote.VerificationSubmitResponseDto) : VerificationUiState()
    data class Error(val message: String) : VerificationUiState()
}

sealed class SharePhotosState {
    object Loading : SharePhotosState()
    data class Success(val photos: List<com.quickpitik.mobile.data.remote.PhotographerLibraryPhotoDto>) : SharePhotosState()
    data class Error(val message: String) : SharePhotosState()
}

sealed class PublicProfileState {
    object Loading : PublicProfileState()
    data class Success(val profile: com.quickpitik.mobile.data.remote.PhotographerProfileDto) : PublicProfileState()
    data class Error(val message: String) : PublicProfileState()
}

sealed class ProfileEventPhotosState {
    object Loading : ProfileEventPhotosState()
    data class Success(val photos: List<com.quickpitik.mobile.data.remote.PhotoDto>) : ProfileEventPhotosState()
    data class Error(val message: String) : ProfileEventPhotosState()
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

    private val _publicEventsState = MutableStateFlow<EventsState>(EventsState.Loading)
    val publicEventsState: StateFlow<EventsState> = _publicEventsState

    private val _activeEvent = MutableStateFlow<PhotographerEventSummaryDto?>(null)
    val activeEvent: StateFlow<PhotographerEventSummaryDto?> = _activeEvent

    private val _queueStats = MutableStateFlow(QueueStats())
    val queueStats: StateFlow<QueueStats> = _queueStats

    private val _earningsUiState = MutableStateFlow<EarningsUiState>(EarningsUiState.Loading)
    val earningsUiState: StateFlow<EarningsUiState> = _earningsUiState

    private val _verificationState = MutableStateFlow<VerificationUiState>(VerificationUiState.Loading)
    val verificationState: StateFlow<VerificationUiState> = _verificationState

    private val _payoutActionState = MutableStateFlow<String?>(null)
    val payoutActionState: StateFlow<String?> = _payoutActionState

    private val _brandSettings = MutableStateFlow<com.quickpitik.mobile.data.remote.BrandSettingsResponseDto?>(null)
    val brandSettings: StateFlow<com.quickpitik.mobile.data.remote.BrandSettingsResponseDto?> = _brandSettings

    private val _payoutAccounts = MutableStateFlow<List<com.quickpitik.mobile.data.remote.PayoutAccountDto>>(emptyList())
    val payoutAccounts: StateFlow<List<com.quickpitik.mobile.data.remote.PayoutAccountDto>> = _payoutAccounts

    private val _socials = MutableStateFlow<List<com.quickpitik.mobile.data.remote.SocialLinkDto>>(emptyList())
    val socials: StateFlow<List<com.quickpitik.mobile.data.remote.SocialLinkDto>> = _socials

    private val _messages = MutableStateFlow<List<com.quickpitik.mobile.data.remote.PhotographerMessageDto>>(emptyList())
    val messages: StateFlow<List<com.quickpitik.mobile.data.remote.PhotographerMessageDto>> = _messages

    private val _sharePhotosState = MutableStateFlow<SharePhotosState>(SharePhotosState.Loading)
    val sharePhotosState: StateFlow<SharePhotosState> = _sharePhotosState

    private val _publicProfileState = MutableStateFlow<PublicProfileState>(PublicProfileState.Loading)
    val publicProfileState: StateFlow<PublicProfileState> = _publicProfileState

    private val _profileEventPhotosState = MutableStateFlow<ProfileEventPhotosState>(ProfileEventPhotosState.Loading)
    val profileEventPhotosState: StateFlow<ProfileEventPhotosState> = _profileEventPhotosState

    init {
        fetchEvents()
        fetchPublicEvents()
        observeQueue()
        fetchEarningsAndTransactions()
        fetchVerificationStatus()
        fetchSettings()
        fetchMessages()
    }

    fun fetchEarningsAndTransactions() {
        viewModelScope.launch {
            _earningsUiState.value = EarningsUiState.Loading
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _earningsUiState.value = EarningsUiState.Error("No valid session. Please log in again.")
                return@launch
            }

            try {
                val overviewResponse = RetrofitClient.apiService.getEarningsOverview("Bearer $token")
                val balanceResponse = RetrofitClient.apiService.getPayoutBalance("Bearer $token")
                val transactionsResponse = RetrofitClient.apiService.getTransactionsLedger("Bearer $token")

                if (overviewResponse.success && overviewResponse.data != null &&
                    balanceResponse.success && balanceResponse.data != null &&
                    transactionsResponse.success && transactionsResponse.data != null
                ) {
                    _earningsUiState.value = EarningsUiState.Success(
                        overview = overviewResponse.data,
                        balance = balanceResponse.data,
                        transactions = transactionsResponse.data.items
                    )
                } else {
                    val errMsg = overviewResponse.error 
                        ?: balanceResponse.error 
                        ?: transactionsResponse.error 
                        ?: "Failed to load financial records."
                    _earningsUiState.value = EarningsUiState.Error(errMsg)
                }
            } catch (e: Exception) {
                _earningsUiState.value = EarningsUiState.Error(e.localizedMessage ?: "Failed to connect to server.")
            }
        }
    }

    fun submitPayoutRequest() {
        viewModelScope.launch {
            _payoutActionState.value = "processing"
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _payoutActionState.value = "Error: No active login session."
                return@launch
            }

            try {
                val response = RetrofitClient.apiService.requestPayout("Bearer $token")
                if (response.success && response.data != null) {
                    _payoutActionState.value = "Success: Payout request submitted successfully!"
                    fetchEarningsAndTransactions()
                } else {
                    _payoutActionState.value = "Error: ${response.error ?: "Payout request rejected."}"
                }
            } catch (e: Exception) {
                _payoutActionState.value = "Error: ${e.localizedMessage ?: "Failed to submit payout request."}"
            }
        }
    }

    fun clearPayoutActionState() {
        _payoutActionState.value = null
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

    fun fetchPublicEvents() {
        viewModelScope.launch {
            _publicEventsState.value = EventsState.Loading
            try {
                val response = RetrofitClient.apiService.getPublicEvents(status = "ACTIVE,COMPLETED,ARCHIVED", limit = 100)
                if (response.success && response.data != null) {
                    val publicList = response.data.items.map { eventDto ->
                        PhotographerEventSummaryDto(
                            id = eventDto.id,
                            slug = eventDto.slug,
                            name = eventDto.name,
                            date = eventDto.date,
                            location = eventDto.location,
                            state = eventDto.status.lowercase(),
                            photoCount = eventDto.photoCount,
                            salesCount = 0,
                            revenueKept = 0.0,
                            bannerUrl = eventDto.bannerUrl
                        )
                    }
                    _publicEventsState.value = EventsState.Success(publicList)
                } else {
                    _publicEventsState.value = EventsState.Error(response.error ?: "Failed to load public events.")
                }
            } catch (e: Exception) {
                _publicEventsState.value = EventsState.Error(e.localizedMessage ?: "Failed to connect to server.")
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

    fun fetchSharePhotos(eventId: String) {
        val token = sessionManager.getAccessToken() ?: return
        viewModelScope.launch {
            _sharePhotosState.value = SharePhotosState.Loading
            try {
                val response = RetrofitClient.apiService.getPhotographerEventPhotos("Bearer $token", eventId)
                if (response.success && response.data != null) {
                    _sharePhotosState.value = SharePhotosState.Success(response.data.items)
                } else {
                    _sharePhotosState.value = SharePhotosState.Error(response.error ?: "Failed to load photos.")
                }
            } catch (e: Exception) {
                _sharePhotosState.value = SharePhotosState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    fun fetchPublicProfile(handle: String) {
        viewModelScope.launch {
            _publicProfileState.value = PublicProfileState.Loading
            try {
                val response = RetrofitClient.apiService.getPublicPhotographerProfile(handle)
                if (response.success && response.data != null) {
                    _publicProfileState.value = PublicProfileState.Success(response.data)
                } else {
                    _publicProfileState.value = PublicProfileState.Error(response.error ?: "Failed to load profile.")
                }
            } catch (e: Exception) {
                _publicProfileState.value = PublicProfileState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    fun fetchProfileEventPhotos(handle: String, slug: String) {
        viewModelScope.launch {
            _profileEventPhotosState.value = ProfileEventPhotosState.Loading
            try {
                val response = RetrofitClient.apiService.getPublicPhotographerEventPhotos(handle, slug)
                if (response.success && response.data != null) {
                    _profileEventPhotosState.value = ProfileEventPhotosState.Success(response.data.items)
                } else {
                    _profileEventPhotosState.value = ProfileEventPhotosState.Error(response.error ?: "Failed to load photos.")
                }
            } catch (e: Exception) {
                _profileEventPhotosState.value = ProfileEventPhotosState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    fun fetchVerificationStatus() {
        viewModelScope.launch {
            _verificationState.value = VerificationUiState.Loading
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _verificationState.value = VerificationUiState.Error("No valid session.")
                return@launch
            }
            try {
                val response = RetrofitClient.apiService.getVerificationStatus("Bearer $token")
                if (response.success && response.data != null) {
                    _verificationState.value = VerificationUiState.Success(response.data)
                } else {
                    _verificationState.value = VerificationUiState.Error(response.error ?: "Failed to fetch onboarding status.")
                }
            } catch (e: Exception) {
                _verificationState.value = VerificationUiState.Error(e.localizedMessage ?: "Failed to load onboarding status.")
            }
        }
    }

    fun fetchMessages() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            try {
                val response = RetrofitClient.apiService.getPhotographerMessages("Bearer $token")
                if (response.success && response.data != null) {
                    _messages.value = response.data
                }
            } catch (e: Exception) {
                // Fail silently
            }
        }
    }

    fun markAllMessagesAsRead() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            try {
                val response = RetrofitClient.apiService.markAllMessagesRead("Bearer $token")
                if (response.success) {
                    fetchMessages()
                }
            } catch (e: Exception) {
                // Fail silently
            }
        }
    }

    fun fetchSettings() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            try {
                val brandResponse = RetrofitClient.apiService.getBrandSettings("Bearer $token")
                if (brandResponse.success && brandResponse.data != null) {
                    _brandSettings.value = brandResponse.data
                }
            } catch (e: Exception) {
                // Fail silently
            }
            try {
                val payoutsResponse = RetrofitClient.apiService.getPayoutAccounts("Bearer $token")
                if (payoutsResponse.success && payoutsResponse.data != null) {
                    _payoutAccounts.value = payoutsResponse.data
                }
            } catch (e: Exception) {
                // Fail silently
            }
            try {
                val socialsResponse = RetrofitClient.apiService.getSocials("Bearer $token")
                if (socialsResponse.success && socialsResponse.data != null) {
                    _socials.value = socialsResponse.data
                }
            } catch (e: Exception) {
                // Fail silently
            }
        }
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

    private val _settingsActionState = MutableStateFlow<String?>(null)
    val settingsActionState: StateFlow<String?> = _settingsActionState

    private val _isSavingSettings = MutableStateFlow(false)
    val isSavingSettings: StateFlow<Boolean> = _isSavingSettings

    fun clearSettingsActionState() {
        _settingsActionState.value = null
    }

    fun saveSettings(
        brandName: String,
        bio: String,
        gcashName: String,
        gcashNumber: String,
        handle: String,
        regionCode: String,
        provinceCode: String,
        socialUrl: String,
        avatarBytes: ByteArray?,
        coverBytes: ByteArray?,
        watermarkBytes: ByteArray?
    ) {
        viewModelScope.launch {
            _isSavingSettings.value = true
            _settingsActionState.value = "Saving settings..."
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _settingsActionState.value = "Error: No valid session. Please log in again."
                _isSavingSettings.value = false
                return@launch
            }

            try {
                // 1. Update Brand (Name & Bio)
                val brandResponse = RetrofitClient.apiService.updateBrand(
                    "Bearer $token",
                    com.quickpitik.mobile.data.remote.BrandPatchRequest(
                        brandName = brandName,
                        brandColor = "none",
                        bio = bio
                    )
                )
                if (!brandResponse.success) {
                    _settingsActionState.value = "Error: " + (brandResponse.error ?: "Failed to update brand.")
                    _isSavingSettings.value = false
                    return@launch
                }

                // 1.1 Update Handle if configured
                if (handle.isNotBlank()) {
                    val handleResponse = RetrofitClient.apiService.updateHandle(
                        "Bearer $token",
                        com.quickpitik.mobile.data.remote.HandlePatchRequest(handle = handle)
                    )
                    if (!handleResponse.success) {
                        _settingsActionState.value = "Error: " + (handleResponse.error ?: "Failed to update handle.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                // 1.2 Update Region if configured
                if (regionCode.isNotBlank() && provinceCode.isNotBlank()) {
                    val regionResponse = RetrofitClient.apiService.updateRegion(
                        "Bearer $token",
                        com.quickpitik.mobile.data.remote.RegionPatchRequest(
                            regionCode = regionCode,
                            provinceCode = provinceCode
                        )
                    )
                    if (!regionResponse.success) {
                        _settingsActionState.value = "Error: " + (regionResponse.error ?: "Failed to update region.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                // 1.3 Update Social Profile link if configured
                if (socialUrl.isNotBlank()) {
                    try {
                        RetrofitClient.apiService.createSocial(
                            "Bearer $token",
                            com.quickpitik.mobile.data.remote.CreateSocialRequest(
                                platform = "facebook",
                                url = socialUrl
                            )
                        )
                    } catch (e: Exception) {
                        // Fail silently if social link already exists
                    }
                }

                // 2. Update GCash Payout
                if (gcashName.isNotBlank() && gcashNumber.isNotBlank()) {
                    val payoutResponse = RetrofitClient.apiService.createPayoutAccount(
                        "Bearer $token",
                        com.quickpitik.mobile.data.remote.CreatePayoutRequest(
                            method = "gcash",
                            accountNumber = gcashNumber,
                            accountName = gcashName
                        )
                    )
                    // Payout endpoint returns 200 OK. Continue.
                }

                // 3. Upload Avatar if chosen
                if (avatarBytes != null) {
                    val requestFile = avatarBytes.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, avatarBytes.size)
                    val part = MultipartBody.Part.createFormData("file", "avatar.jpg", requestFile)
                    val avatarResponse = RetrofitClient.apiService.uploadAvatar("Bearer $token", part)
                    if (!avatarResponse.success) {
                        _settingsActionState.value = "Error: " + (avatarResponse.error ?: "Failed to upload avatar.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                // 4. Upload Cover if chosen
                if (coverBytes != null) {
                    val requestFile = coverBytes.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, coverBytes.size)
                    val part = MultipartBody.Part.createFormData("file", "cover.jpg", requestFile)
                    val coverResponse = RetrofitClient.apiService.uploadCover("Bearer $token", part)
                    if (!coverResponse.success) {
                        _settingsActionState.value = "Error: " + (coverResponse.error ?: "Failed to upload cover.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                // 5. Upload Watermark if chosen
                if (watermarkBytes != null) {
                    val requestFile = watermarkBytes.toRequestBody("image/png".toMediaTypeOrNull(), 0, watermarkBytes.size)
                    val part = MultipartBody.Part.createFormData("file", "watermark.png", requestFile)
                    val watermarkResponse = RetrofitClient.apiService.uploadWatermark("Bearer $token", part)
                    if (!watermarkResponse.success) {
                        _settingsActionState.value = "Error: " + (watermarkResponse.error ?: "Failed to upload watermark.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                _settingsActionState.value = "Success: Settings updated successfully!"
                fetchVerificationStatus()
                fetchSettings()
            } catch (e: Exception) {
                _settingsActionState.value = "Error: " + (e.localizedMessage ?: "Connection error.")
            } finally {
                _isSavingSettings.value = false
            }
        }
    }

    fun submitVerification() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            if (token == null) return@launch
            _settingsActionState.value = "Submitting verification..."
            try {
                val response = RetrofitClient.apiService.submitVerification("Bearer $token")
                if (response.success && response.data != null) {
                    _verificationState.value = VerificationUiState.Success(response.data)
                    _settingsActionState.value = "Success: Submitted for admin review!"
                } else {
                    _settingsActionState.value = "Error: " + (response.error ?: "Failed to submit verification.")
                }
            } catch (e: Exception) {
                _settingsActionState.value = "Error: " + (e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun withdrawVerification() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            if (token == null) return@launch
            _settingsActionState.value = "Withdrawing verification..."
            try {
                val response = RetrofitClient.apiService.withdrawVerification("Bearer $token")
                if (response.success && response.data != null) {
                    _verificationState.value = VerificationUiState.Success(response.data)
                    _settingsActionState.value = "Success: Verification review rescinded."
                } else {
                    _settingsActionState.value = "Error: " + (response.error ?: "Failed to withdraw verification.")
                }
            } catch (e: Exception) {
                _settingsActionState.value = "Error: " + (e.localizedMessage ?: "Connection error.")
            }
        }
    }
}

