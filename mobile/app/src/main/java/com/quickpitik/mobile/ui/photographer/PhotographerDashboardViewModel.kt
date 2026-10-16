package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.repository.PhotoMetadataRepository
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.data.remote.PhotographerMessageFrame
import com.quickpitik.mobile.data.remote.QpWebSocket
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.apiCall
import com.quickpitik.mobile.data.remote.uiMessage
import com.quickpitik.mobile.data.remote.WsState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody


sealed class EventsState {
    object Loading : EventsState()
    data class Success(val events: List<PhotographerEventSummaryDto>) : EventsState()
    data class Error(val message: String) : EventsState()
}

sealed class VerificationUiState {
    object Loading : VerificationUiState()
    data class Success(val verification: com.quickpitik.mobile.data.remote.VerificationSubmitResponseDto) : VerificationUiState()
    data class Error(val message: String) : VerificationUiState()
}

// Photographer-owned event authoring (V46).
sealed class EventFormState {
    object Idle : EventFormState()
    object Submitting : EventFormState()
    data class Success(val event: com.quickpitik.mobile.data.remote.PhotographerEventDetailDto) : EventFormState()
    data class Error(val message: String) : EventFormState()
}

sealed class CouponUiState {
    object Loading : CouponUiState()
    // coupon == null → none set yet (backend answers success + null data).
    data class Loaded(val coupon: com.quickpitik.mobile.data.remote.CouponDto?) : CouponUiState()
    data class Error(val message: String) : CouponUiState()
}

// Only-changed fields are non-null (create sends all; edit diffs). Strings map
// straight to the multipart @RequestParam parts; coverUri is a content:// Uri.
data class OwnedEventFormInput(
    val title: String? = null,
    val date: String? = null,
    val location: String? = null,
    val organizerName: String? = null,
    val description: String? = null,
    val visibility: String? = null,
    val pricingMode: String? = null,
    val pricePerPhoto: String? = null,
    val watermarkPolicy: String? = null,
    val withdrawPendingChange: Boolean = false,
    val coverUri: String? = null,
)

sealed class SharePhotosState {
    object Loading : SharePhotosState()
    data class Success(
        val photos: List<com.quickpitik.mobile.data.remote.PhotographerLibraryPhotoDto>,
        val total: Long = photos.size.toLong(),
        val nextOffset: Int = photos.size,
        val loadingMore: Boolean = false,
        val loadMoreError: String? = null,
    ) : SharePhotosState() {
        val hasMore: Boolean get() = nextOffset < total
    }
    data class Error(val message: String) : SharePhotosState()
}

class PhotographerDashboardViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)
    private val metadataRepository = PhotoMetadataRepository(sessionManager)
    @Volatile private var metadataRequest = 0
    private val _eventMetadataVisibility = MutableStateFlow(MetadataToggleState())
    val eventMetadataVisibility: StateFlow<MetadataToggleState> = _eventMetadataVisibility

    private fun metadataContext(): Pair<SessionManager.AuthSession, String>? =
        sessionManager.getAccessToken()?.let { token ->
            sessionManager.authSnapshot(token)?.let { it to RetrofitClient.BASE_URL }
        }

    private fun current(context: Pair<SessionManager.AuthSession, String>, eventId: String) =
        sessionManager.isCurrent(context.first) && context.second == RetrofitClient.BASE_URL &&
        _eventMetadataVisibility.value.eventId == eventId

    fun loadEventMetadataVisibility(eventId: String) {
        _eventMetadataVisibility.value = MetadataToggleState(eventId = eventId)
        val context = metadataContext() ?: return
        val request = ++metadataRequest
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val saved = metadataRepository.getEventVisibility(eventId).metadataPublic
                if (current(context, eventId) && request == metadataRequest) _eventMetadataVisibility.value = MetadataToggleState(eventId, saved, saved, loading = false)
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                if (current(context, eventId) && request == metadataRequest) _eventMetadataVisibility.value = MetadataToggleState(eventId = eventId, loading = false, error = "Couldn't load metadata visibility.")
            }
        }
    }

    fun setEventMetadataVisibilityDraft(value: Boolean) {
        val state = _eventMetadataVisibility.value
        if (!state.loading && !state.saving && state.saved != null) _eventMetadataVisibility.value = state.copy(draft = value, error = null)
    }

    fun cancelEventMetadataVisibility() {
        val state = _eventMetadataVisibility.value
        if (!state.saving) _eventMetadataVisibility.value = state.copy(draft = state.saved, error = null)
    }

    fun saveEventMetadataVisibility() {
        val state = _eventMetadataVisibility.value
        val eventId = state.eventId ?: return
        val value = state.draft ?: return
        if (state.loading || state.saving || value == state.saved) return
        val context = metadataContext() ?: return
        val request = metadataRequest
        _eventMetadataVisibility.value = state.copy(saving = true, error = null)
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val saved = metadataRepository.setEventVisibility(eventId, value).metadataPublic
                if (current(context, eventId) && request == metadataRequest) _eventMetadataVisibility.value = MetadataToggleState(eventId, saved, saved, loading = false)
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                if (current(context, eventId) && request == metadataRequest) _eventMetadataVisibility.value = _eventMetadataVisibility.value.copy(saving = false, error = "Couldn't save metadata visibility.")
            }
        }
    }
    private val _eventsState = MutableStateFlow<EventsState>(EventsState.Loading)
    val eventsState: StateFlow<EventsState> = _eventsState

    private val _publicEventsState = MutableStateFlow<EventsState>(EventsState.Loading)
    val publicEventsState: StateFlow<EventsState> = _publicEventsState

    private val _verificationState = MutableStateFlow<VerificationUiState>(VerificationUiState.Loading)
    val verificationState: StateFlow<VerificationUiState> = _verificationState

    // The single dashboard railguard signal. Every restricted action (create
    // event/coupon, sync, share, profile access) reads this instead of
    // re-deriving the approved check, so the rule lives in one place and future
    // dashboard features inherit it. Fail-open per the 2026-09-03 ADR: an unknown
    // / still-loading status never blocks — only a loaded, non-approved status
    // does; the backend (PhotographerAccessGuard) is the real enforcer.
    val dashboardApproved: StateFlow<Boolean> = verificationState
        .map { st ->
            val status = (st as? VerificationUiState.Success)?.verification?.status?.lowercase()
            status == null || status == "approved"
        }
        .stateIn(viewModelScope, SharingStarted.Eagerly, true)

    // Owned-event authoring (V46).
    private val _eventFormState = MutableStateFlow<EventFormState>(EventFormState.Idle)
    val eventFormState: StateFlow<EventFormState> = _eventFormState
    // Edit prefill: full detail (description/organizer/price) the list summary lacks.
    private val _eventEditTarget = MutableStateFlow<com.quickpitik.mobile.data.remote.PhotographerEventDetailDto?>(null)
    val eventEditTarget: StateFlow<com.quickpitik.mobile.data.remote.PhotographerEventDetailDto?> = _eventEditTarget
    private val _couponState = MutableStateFlow<CouponUiState>(CouponUiState.Loading)
    val couponState: StateFlow<CouponUiState> = _couponState
    private val _couponActionMessage = MutableStateFlow<String?>(null)
    val couponActionMessage: StateFlow<String?> = _couponActionMessage

    // Result of Submit / Withdraw on the settings screen's verification card.
    // Its own channel, not the settings one: this ViewModel owns the status
    // those two actions change, so it owns what they report.
    private val _verificationActionState = MutableStateFlow<ActionState>(ActionState.Idle)
    val verificationActionState: StateFlow<ActionState> = _verificationActionState

    private val _messages = MutableStateFlow<List<com.quickpitik.mobile.data.remote.PhotographerMessageDto>>(emptyList())
    val messages: StateFlow<List<com.quickpitik.mobile.data.remote.PhotographerMessageDto>> = _messages

    // Inbox push channel. Same shape as the runner inbox in
    // RunnerInboxViewModel — the two were structurally identical on the refetch
    // path and stay identical here.
    private val inboxSocket = QpWebSocket(sessionManager, viewModelScope)
    private val inboxGson = com.google.gson.Gson()

    private val _sharePhotosState = MutableStateFlow<SharePhotosState>(SharePhotosState.Loading)
    val sharePhotosState: StateFlow<SharePhotosState> = _sharePhotosState

    init {
        fetchEvents()
        fetchVerificationStatus()
        observeInboxPush()
    }

    /**
     * Bell parity with the website's photographer notification channel: an
     * admin approving a verification or resolving a dispute reaches the shade
     * badge without the photographer navigating anywhere.
     *
     * [connectInbox] / [disconnectInbox] are driven by the screen's lifecycle,
     * not from here, so a backgrounded app isn't holding a socket open.
     */
    private fun observeInboxPush() {
        viewModelScope.launch {
            inboxSocket.state.collect { state ->
                // Every (re)open refetches: anything pushed while the socket was
                // down was missed outright. Matches the website's onopen grace.
                if (state is WsState.Open) fetchMessages()
            }
        }
        viewModelScope.launch {
            inboxSocket.frames.collect { raw ->
                val frame = runCatching {
                    inboxGson.fromJson(raw, PhotographerMessageFrame::class.java)
                }.getOrNull()
                if (frame?.type != "message.created") return@collect
                val message = frame.message ?: return@collect
                val current = _messages.value
                if (current.any { it.id == message.id }) return@collect
                _messages.value = listOf(message) + current
            }
        }
    }

    fun connectInbox() = inboxSocket.connect(INBOX_CHANNEL)

    fun disconnectInbox() = inboxSocket.close()

    override fun onCleared() {
        inboxSocket.release()
        super.onCleared()
    }

    // Returns the Job so pull-to-refresh can join() it (fire-and-forget
    // callers unaffected).
    fun fetchEvents(): Job = viewModelScope.launch {
            val cached = _eventsState.value as? EventsState.Success
            if (cached == null) _eventsState.value = EventsState.Loading
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _eventsState.value = EventsState.Error("No valid session. Please log in again.")
                return@launch
            }

            try {
                val response = RetrofitClient.apiService.getPhotographerEvents("Bearer $token")
                if (response.success && response.data != null) {
                    _eventsState.value = EventsState.Success(response.data.items)
                } else {
                    _eventsState.value = cached ?: EventsState.Error(response.error ?: "Failed to load events.")
                }
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                _eventsState.value = cached ?: EventsState.Error(e.localizedMessage ?: "Failed to connect to server.")
            }
        }

    fun fetchPublicEvents(): Job = viewModelScope.launch {
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

    // --- Owned-event authoring (V46) ---

    // eventId == null → create (POST); else edit (PATCH, only the non-null fields
    // in `input` are sent). On success refetch the list so review chips update.
    fun submitEvent(eventId: String?, input: OwnedEventFormInput) {
        // IO: buildEventParts reads the cover image bytes off the main
        // thread, as every upload path in this app does.
        viewModelScope.launch(Dispatchers.IO) {
            _eventFormState.value = EventFormState.Submitting
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _eventFormState.value = EventFormState.Error("No valid session. Please log in again.")
                return@launch
            }
            // buildEventParts stays inside the lambda: a cover image that
            // cannot be read must reach the same Error copy the request does.
            apiCall("Couldn't save the event.") {
                val parts = buildEventParts(input)
                if (eventId == null) {
                    RetrofitClient.apiService.createPhotographerEvent("Bearer $token", parts)
                } else {
                    RetrofitClient.apiService.updatePhotographerEvent("Bearer $token", eventId, parts)
                }
            }.onSuccess {
                _eventFormState.value = EventFormState.Success(it)
                fetchEvents()
            }.onFailure {
                _eventFormState.value = EventFormState.Error(it.uiMessage)
            }
        }
    }

    private fun buildEventParts(input: OwnedEventFormInput): List<MultipartBody.Part> {
        val parts = mutableListOf<MultipartBody.Part>()
        fun text(name: String, value: String?) {
            if (value != null) parts.add(MultipartBody.Part.createFormData(name, value))
        }
        text("title", input.title)
        text("date", input.date)
        text("location", input.location)
        text("organizerName", input.organizerName)
        text("description", input.description)
        text("visibility", input.visibility)
        text("pricingMode", input.pricingMode)
        text("pricePerPhoto", input.pricePerPhoto)
        text("watermarkPolicy", input.watermarkPolicy)
        if (input.withdrawPendingChange) {
            parts.add(MultipartBody.Part.createFormData("withdrawPendingChange", "true"))
        }
        input.coverUri?.let { uri ->
            val bytes = getApplication<Application>().contentResolver
                .openInputStream(android.net.Uri.parse(uri))?.use { it.readBytes() }
            if (bytes != null) {
                val body = bytes.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, bytes.size)
                parts.add(MultipartBody.Part.createFormData("cover", "cover.jpg", body))
            }
        }
        return parts
    }

    fun loadEventDetail(eventId: String) {
        viewModelScope.launch {
            _eventEditTarget.value = null
            val token = sessionManager.getAccessToken() ?: return@launch
            try {
                val response = RetrofitClient.apiService.getPhotographerEventDetail("Bearer $token", eventId)
                if (response.success && response.data != null) _eventEditTarget.value = response.data
            } catch (_: Exception) {
                // Non-fatal: the form falls back to the list-summary prefill.
            }
        }
    }

    fun resetEventForm() {
        _eventFormState.value = EventFormState.Idle
        _eventEditTarget.value = null
    }

    fun loadCoupon(eventId: String) {
        viewModelScope.launch {
            _couponState.value = CouponUiState.Loading
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _couponState.value = CouponUiState.Error("No valid session.")
                return@launch
            }
            try {
                val response = RetrofitClient.apiService.getEventCoupon("Bearer $token", eventId)
                _couponState.value = CouponUiState.Loaded(response.data)
            } catch (e: Exception) {
                _couponState.value = CouponUiState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    fun saveCoupon(
        eventId: String,
        code: String,
        percentOff: Int,
        active: Boolean,
        expiresAt: String?,
        usageLimit: Int?,
    ) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            apiCall("Couldn't save the coupon.") {
                RetrofitClient.apiService.upsertEventCoupon(
                    "Bearer $token",
                    eventId,
                    com.quickpitik.mobile.data.remote.UpsertCouponRequest(
                        code = code,
                        percentOff = percentOff,
                        active = active,
                        expiresAt = expiresAt,
                        usageLimit = usageLimit,
                    ),
                )
            }.onSuccess {
                _couponState.value = CouponUiState.Loaded(it)
            }.onFailure {
                _couponActionMessage.value = it.uiMessage
            }
        }
    }

    fun deleteCoupon(eventId: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            try {
                RetrofitClient.apiService.deleteEventCoupon("Bearer $token", eventId)
                _couponState.value = CouponUiState.Loaded(null)
            } catch (e: Exception) {
                _couponActionMessage.value = RetrofitClient.parseError(e)
            }
        }
    }

    fun clearCouponMessage() { _couponActionMessage.value = null }

    fun fetchSharePhotos(eventId: String, loadMore: Boolean = false) =
        fetchPhotoPage(eventId, deleted = false, loadMore = loadMore)

    /**
     * Resolves the presigned URL for the photographer's own un-watermarked
     * original. Suspending rather than state-backed: the share page needs the
     * link for exactly one download tap, so parking it in a StateFlow would
     * leave a stale presigned URL sitting in memory after it expires.
     */
    suspend fun resolvePhotoDownloadUrl(photoId: String): Result<String> {
        val token = sessionManager.getAccessToken()
            ?: return Result.failure(IllegalStateException("You're signed out. Sign in and try again."))
        return try {
            val response = RetrofitClient.apiService
                .getPhotographerPhotoDownload("Bearer $token", photoId)
            val url = response.data?.url
            if (response.success && url != null) {
                Result.success(url)
            } else {
                Result.failure(IllegalStateException(response.error ?: "Couldn't get the download link."))
            }
        } catch (e: Exception) {
            Result.failure(IllegalStateException(RetrofitClient.parseError(e)))
        }
    }

    // ── Photo library management (V52): Deleted tab + trash/restore/permanent ──
    private val _trashPhotosState = MutableStateFlow<SharePhotosState>(SharePhotosState.Loading)
    val trashPhotosState: StateFlow<SharePhotosState> = _trashPhotosState

    private val photoPageJobs = mutableMapOf<Boolean, Job>()
    private val photoPageEvents = mutableMapOf<Boolean, String>()

    fun fetchTrashPhotos(eventId: String, loadMore: Boolean = false) =
        fetchPhotoPage(eventId, deleted = true, loadMore = loadMore)

    private fun fetchPhotoPage(eventId: String, deleted: Boolean, loadMore: Boolean) {
        val token = sessionManager.getAccessToken() ?: return
        val state = if (deleted) _trashPhotosState else _sharePhotosState
        val previous = if (loadMore && photoPageEvents[deleted] == eventId) state.value as? SharePhotosState.Success else null
        if (loadMore && (previous == null || !previous.hasMore || previous.loadingMore)) return
        photoPageJobs[deleted]?.cancel()
        photoPageEvents[deleted] = eventId
        state.value = previous?.copy(loadingMore = true, loadMoreError = null) ?: SharePhotosState.Loading
        photoPageJobs[deleted] = viewModelScope.launch {
            try {
                val response = RetrofitClient.apiService
                    .getPhotographerEventPhotos("Bearer $token", eventId,
                        offset = previous?.nextOffset ?: 0, deleted = deleted)
                if (response.success && response.data != null) {
                    val page = response.data
                    state.value = SharePhotosState.Success(
                        photos = ((previous?.photos ?: emptyList()) + page.items).distinctBy { it.id },
                        total = page.total,
                        nextOffset = if (page.items.isEmpty()) page.total.toInt() else page.offset + page.items.size,
                    )
                } else {
                    throw IllegalStateException(response.error ?: "Failed to load photos.")
                }
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                val message = RetrofitClient.parseError(e)
                state.value = previous?.copy(loadingMore = false, loadMoreError = message) ?: SharePhotosState.Error(message)
            }
        }
    }

    // Trash / restore are bulk; the per-photo UI sends a one-element list.
    // Returns the affected count; refresh also runs after a partial batch failure.
    suspend fun trashPhotos(eventId: String, photoIds: List<String>): Result<Int> =
        movePhotos(eventId, photoIds, restore = false)

    suspend fun restorePhotos(eventId: String, photoIds: List<String>): Result<Int> =
        movePhotos(eventId, photoIds, restore = true)

    private suspend fun movePhotos(eventId: String, photoIds: List<String>, restore: Boolean): Result<Int> =
        mutatePhotos(eventId) { token ->
            var affected = 0
            try {
                for (batch in photoIds.distinct().chunked(500)) {
                    val request = com.quickpitik.mobile.data.remote.PhotoIdsRequest(batch)
                    val response = if (restore) RetrofitClient.apiService.restorePhotographerPhotos("Bearer $token", eventId, request)
                        else RetrofitClient.apiService.trashPhotographerPhotos("Bearer $token", eventId, request)
                    if (!response.success || response.data == null) error(response.error ?: "Photo action failed")
                    affected += response.data.affected
                }
                Result.success(affected)
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                Result.failure(IllegalStateException("$affected photos updated. Couldn't finish the remaining photos; refresh or retry."))
            }
        }

    suspend fun permanentDeletePhotos(
        eventId: String,
        photoIds: List<String>,
    ): Result<com.quickpitik.mobile.data.remote.PermanentDeleteResponseDto> =
        mutatePhotos(eventId) { token ->
            val deleted = mutableListOf<String>()
            val sold = mutableListOf<String>()
            val pending = mutableListOf<String>()
            val failed = mutableListOf<String>()
            for (batch in photoIds.distinct().chunked(100)) {
                try {
                    val response = RetrofitClient.apiService.permanentDeletePhotographerPhotos(
                        "Bearer $token", eventId, com.quickpitik.mobile.data.remote.PhotoIdsRequest(batch))
                    if (!response.success || response.data == null) error(response.error ?: "Photo deletion failed")
                    val result = response.data
                    deleted += result.deleted
                    sold += result.blockedSold
                    pending += result.blockedPending
                    failed += batch - (result.deleted + result.blockedSold + result.blockedPending).toSet()
                } catch (e: Exception) {
                    if (e is kotlinx.coroutines.CancellationException) throw e
                    failed += batch
                }
            }
            Result.success(com.quickpitik.mobile.data.remote.PermanentDeleteResponseDto(deleted, sold, pending, failed))
        }

    // Refresh both photo lists and event counts after any mutation attempt.
    private suspend fun <T> mutatePhotos(
        eventId: String,
        call: suspend (String) -> Result<T>,
    ): Result<T> {
        val token = sessionManager.getAccessToken()
            ?: return Result.failure(IllegalStateException("You're signed out. Sign in and try again."))
        return try {
            call(token)
        } catch (e: Exception) {
            if (e is kotlinx.coroutines.CancellationException) throw e
            Result.failure(IllegalStateException(RetrofitClient.parseError(e)))
        } finally {
            // A later batch can fail after earlier batches committed.
            fetchSharePhotos(eventId)
            fetchTrashPhotos(eventId)
            fetchEvents()
        }
    }

    // "Download all" — mints the tokened ZIP URL; the screen streams it into
    // Downloads. Suspending like resolvePhotoDownloadUrl (consumed once).
    suspend fun prepareBundleUrl(eventId: String): Result<String> {
        val token = sessionManager.getAccessToken()
            ?: return Result.failure(IllegalStateException("You're signed out. Sign in and try again."))
        return try {
            val response = RetrofitClient.apiService.preparePhotographerBundle("Bearer $token", eventId)
            val url = response.data?.url
            if (response.success && url != null) Result.success(url)
            else Result.failure(IllegalStateException(response.error ?: "Couldn't prepare the download."))
        } catch (e: Exception) {
            Result.failure(IllegalStateException(RetrofitClient.parseError(e)))
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

    fun markMessageAsRead(messageId: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            try {
                val response = RetrofitClient.apiService.markMessageRead("Bearer $token", messageId)
                if (response.success) {
                    fetchMessages()
                }
            } catch (e: Exception) {
                // Fail silently
            }
        }
    }

    fun removeMessage(messageId: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            try {
                val response = RetrofitClient.apiService.removePhotographerMessage("Bearer $token", messageId)
                if (response.success) {
                    fetchMessages()
                }
            } catch (e: Exception) {
                // Fail silently
            }
        }
    }

    fun clearVerificationActionState() {
        _verificationActionState.value = ActionState.Idle
    }

    fun submitVerification() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            if (token == null) return@launch
            _verificationActionState.value = ActionState.Busy("Submitting verification...")
            try {
                val response = RetrofitClient.apiService.submitVerification("Bearer $token")
                if (response.success && response.data != null) {
                    // Straight from the response, not a refetch: a refetch would
                    // flip the card back to Loading for a frame.
                    _verificationState.value = VerificationUiState.Success(response.data)
                    _verificationActionState.value = ActionState.Done("Submitted for admin review!")
                } else {
                    _verificationActionState.value =
                        ActionState.Failed(response.error ?: "Failed to submit verification.")
                }
            } catch (e: Exception) {
                _verificationActionState.value =
                    ActionState.Failed(e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun withdrawVerification() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            if (token == null) return@launch
            _verificationActionState.value = ActionState.Busy("Withdrawing verification...")
            try {
                val response = RetrofitClient.apiService.withdrawVerification("Bearer $token")
                if (response.success && response.data != null) {
                    _verificationState.value = VerificationUiState.Success(response.data)
                    _verificationActionState.value = ActionState.Done("Verification review rescinded.")
                } else {
                    _verificationActionState.value =
                        ActionState.Failed(response.error ?: "Failed to withdraw verification.")
                }
            } catch (e: Exception) {
                _verificationActionState.value =
                    ActionState.Failed(e.localizedMessage ?: "Connection error.")
            }
        }
    }

    private companion object {
        const val INBOX_CHANNEL = "/ws/me/photographer/notifications"
    }
}

