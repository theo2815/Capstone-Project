package com.quickpitik.mobile.ui.photographer

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.BuildConfig
import com.quickpitik.mobile.data.MAX_UPLOAD_BYTES
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.StatusCount
import com.quickpitik.mobile.data.local.TetherEvents
import com.quickpitik.mobile.data.local.TetherWatchPrefs
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.data.local.UploadSpool
import com.quickpitik.mobile.data.local.TetherStorageCleanup
import com.quickpitik.mobile.data.readAtMost
import com.quickpitik.mobile.data.remote.EarningsOverviewDto
import com.quickpitik.mobile.data.remote.PayoutBalanceDto
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.data.remote.PhotographerMessageFrame
import com.quickpitik.mobile.data.remote.PhotographerTransactionDto
import com.quickpitik.mobile.data.remote.QpWebSocket
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.WsState
import com.quickpitik.mobile.data.usb.CameraIngestCoordinator
import com.quickpitik.mobile.data.usb.ShutterWatchState
import com.quickpitik.mobile.worker.PhotoUploadScheduler
import com.quickpitik.mobile.data.usb.CameraConnectionState
import com.quickpitik.mobile.data.usb.ptp.CardPhoto
import com.quickpitik.mobile.data.usb.ptp.UsbCardBrowseController
import com.quickpitik.mobile.data.usb.ptp.UsbCardImportController
import com.quickpitik.mobile.ui.common.canUploadToEvent
import com.quickpitik.mobile.worker.PhotoUploadWorker
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.SharingStarted
import com.quickpitik.mobile.data.local.CameraRecoverySummary
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale


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
        val transactions: List<PhotographerTransactionDto>,
        // Server-computed per-month totals over ALL transactions (not just the
        // loaded page) — drives the ledger's month subtotals.
        val monthTotals: Map<String, Double> = emptyMap(),
    ) : EarningsUiState()
    data class Error(val message: String) : EarningsUiState()
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

data class QueueStats(
    val syncedCount: Int = 0,
    val queuedCount: Int = 0,
    val uploadingCount: Int = 0,
    val failedCount: Int = 0,
    /** FAILED rows that are "already in another event" refusals; the Remove duplicates action settles them. */
    val duplicateCount: Int = 0,
    val totalCount: Int = 0,
    val progress: Float = 0f,
    /**
     * Most recent error message attached to a FAILED row, or null when nothing
     * has failed. PhotoUploadWorker already writes a per-row string into
     * UploadRecord.errorMessage; without surfacing it here the UI was stuck
     * showing "Failed · N" with no detail, which made the 46-failure batch on
     * 2026-05-28 impossible to diagnose from the screen alone.
     */
    val lastError: String? = null,
    /**
     * Full queue rows, newest first, for the per-photo sync strip. The Flow
     * already delivered these — they were previously reduced to counts only.
     */
    val recentRecords: List<com.quickpitik.mobile.data.local.UploadRecord> = emptyList(),
)

private val DUPLICATE_IN_EVENT = Regex("""^This photo already exists in your event '(.*)'\.$""")
private val NETWORK_HINTS = listOf("upload interrupted", "timeout", "timed out", "unable to resolve host",
    "failed to connect", "connection abort", "connection reset", "network is unreachable")

/**
 * Plain wording for the sync card's latest failure. Rows store the backend
 * message or the exception text, never the code, so this matches wording;
 * anything unrecognised passes through unchanged.
 */
// ponytail: matches backend wording; store the error code on the row if the backend copy starts drifting.
internal fun friendlyUploadError(raw: String): String {
    DUPLICATE_IN_EVENT.find(raw)?.let { match ->
        val event = match.groupValues[1]
        return if (event == "another event") "This photo is already in another of your events, so it wasn't added here. Nothing to fix."
            else "This photo is already in your event \"$event\", so it wasn't added here. Nothing to fix."
    }
    val lower = raw.lowercase()
    return when {
        raw == "This event is not accepting uploads." -> "This event isn't accepting uploads right now. Check its dates or status."
        raw == "This event belongs to another photographer." -> "This event belongs to another photographer, so photos can't be added to it."
        raw.startsWith("Upload must be image/") || raw.startsWith("Image dimensions exceed") ->
            "This file can't be uploaded. Use a JPEG, PNG or WebP under 25 MB."
        raw.startsWith("The uploaded file wasn't found in storage") -> "The upload didn't finish. Tap Retry to send it again."
        raw.startsWith("Storage upload returned HTTP") -> "The photo didn't reach cloud storage. Tap Retry to send it again."
        raw == "Upload rejected by server." -> "QuickPitik didn't accept this photo. Tap Retry. If it keeps happening, contact support."
        NETWORK_HINTS.any { it in lower } -> "Couldn't reach QuickPitik. Check your connection. Photos retry on their own."
        else -> raw
    }
}

/**
 * UI-side state for the manual "Import from camera card" flow (Increment 1 of
 * the camera-import series). `Idle` keeps the sheet hidden; any non-Idle value
 * mounts [com.quickpitik.mobile.ui.photographer.CameraCardImportSheet].
 */
sealed class CardBrowseState {
    object Idle : CardBrowseState()
    object Opening : CardBrowseState()
    data class Scanning(val seen: Int, val total: Int) : CardBrowseState()
    /**
     * Card walked — `photos` is what the photographer can import; `selectedHandles`
     * is which of them they've ticked (Increment 2). `importedHandles` is the
     * subset of `photos` that succeeded in any earlier import this VM lifetime
     * (Increment 4 D dedupe) — rows render as "Imported" and are excluded from
     * selection.
     */
    data class Loaded(
        val photos: List<CardPhoto>,
        val selectedHandles: Set<Long> = emptySet(),
        val importedHandles: Set<Long> = emptySet(),
        // Bumped on every progressive-thumbnail batch. CardPhoto equality is
        // by handle (deliberately), so without this a Loaded carrying fresh
        // thumbnail bytes would compare equal and the StateFlow would drop it.
        val thumbsVersion: Int = 0,
    ) : CardBrowseState()
    /**
     * Bytes flowing off the card. Counts come from the underlying handle sets
     * (the controller emits Sets so the VM can drive dedupe + retry-failed).
     */
    data class Importing(
        val seen: Int,
        val total: Int,
        val succeeded: Int,
        val failed: Int,
    ) : CardBrowseState()
    /**
     * All selected handles attempted — sheet shows a summary + Done. Carries
     * `photos` + `failedHandles` so the sheet can offer a "Retry N failed"
     * (Increment 4 B) without re-browsing the card.
     */
    data class ImportDone(
        val photos: List<CardPhoto>,
        val succeededHandles: Set<Long>,
        val failedHandles: Set<Long>,
    ) : CardBrowseState()
    data class Error(val message: String) : CardBrowseState()
}

// Newest rows the Capture tab observes for its strip (the screen shows ≤30).
private const val RECENT_RECORDS_LIMIT = 200

class PhotographerDashboardViewModel(application: Application) : AndroidViewModel(application) {
    private val database = AppDatabase.getDatabase(application)
    private val sessionManager = SessionManager.getInstance(application)
    private val coordinator = CameraIngestCoordinator.get(application)
    private val cameraManager = coordinator.cameraManager
    val cameraConnectionState: StateFlow<CameraConnectionState> = cameraManager.state

    val phoneSpace = UploadSpool.observeSpace(application)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), UploadSpool.Space())
    data class CleanupState(
        val review: TetherStorageCleanup.Review? = null,
        val busy: Boolean = false,
        val message: String? = null,
    )
    private val cleanup = TetherStorageCleanup(application, database)
    private val _cleanupState = MutableStateFlow(CleanupState())
    val cleanupState: StateFlow<CleanupState> = _cleanupState
    private var cleanupJob: Job? = null

    fun reviewPhoneCleanup() {
        if (_cleanupState.value.busy) return
        _cleanupState.value = CleanupState(busy = true)
        cleanupJob = viewModelScope.launch {
            try {
                val review = cleanup.review()
                _cleanupState.value = if (review.records.isEmpty()) CleanupState(message =
                    "No uploaded tether originals are ready to check. Pending, failed and older unverified copies stay on this phone.")
                else CleanupState(review = review)
            } catch (e: CancellationException) { throw e }
            catch (_: Exception) { _cleanupState.value = CleanupState(message = "Couldn't review phone copies. Try again.") }
        }
    }

    fun cancelPhoneCleanup() {
        cleanupJob?.cancel()
        _cleanupState.value = CleanupState(message = "Cleanup stopped. Unchecked copies stay on this phone.")
    }

    fun confirmPhoneCleanup() {
        val review = _cleanupState.value.review ?: return
        if (_cleanupState.value.busy) return
        _cleanupState.value = CleanupState(busy = true)
        cleanupJob = viewModelScope.launch {
            var removed = 0
            var freed = 0L
            var kept = 0
            var historyFailed = false
            for (record in review.records) {
                try {
                    val result = cleanup.remove(record)
                    freed += result.bytes
                    removed++
                    if (!result.recorded) {
                        historyFailed = true
                        kept = review.records.size - removed
                        break
                    }
                }
                catch (e: CancellationException) { throw e }
                catch (_: java.io.IOException) { kept = review.records.size - removed; break }
                catch (_: Exception) { kept++ }
                _cleanupState.value = CleanupState(busy = true,
                    message = "Checked ${removed + kept} of ${review.records.size} copies · $removed removed")
            }
            _cleanupState.value = CleanupState(message = "$removed phone copies removed · " +
                android.text.format.Formatter.formatShortFileSize(getApplication(), freed) + " freed." +
                (if (kept > 0) " $kept kept because verification or removal did not complete. You can try again." else "") +
                if (historyFailed) " Cleanup stopped because local history couldn't be updated." else "")
        }
    }

    private val _eventsState = MutableStateFlow<EventsState>(EventsState.Loading)
    val eventsState: StateFlow<EventsState> = _eventsState

    private val _publicEventsState = MutableStateFlow<EventsState>(EventsState.Loading)
    val publicEventsState: StateFlow<EventsState> = _publicEventsState

    private val _activeEvent = MutableStateFlow<PhotographerEventSummaryDto?>(null)
    val activeEvent: StateFlow<PhotographerEventSummaryDto?> = _activeEvent
    @OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
    val cameraRecovery: StateFlow<CameraRecoverySummary> = _activeEvent.flatMapLatest { event ->
        if (event == null) flowOf(CameraRecoverySummary())
        else database.cameraTransferDao().observeRecovery(sessionManager.getUserEmail().orEmpty(), event.id)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), CameraRecoverySummary())

    fun retryCameraTransfers() {
        val event = _activeEvent.value ?: return
        val owner = sessionManager.getUserEmail() ?: return
        viewModelScope.launch {
            database.cameraTransferDao().retry(owner, event.id)
            if (!coordinator.isWatching) startShutterWatch()
        }
    }


    private val _queueStats = MutableStateFlow(QueueStats())
    val queueStats: StateFlow<QueueStats> = _queueStats

    private val _earningsUiState = MutableStateFlow<EarningsUiState>(EarningsUiState.Loading)
    val earningsUiState: StateFlow<EarningsUiState> = _earningsUiState

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

    private val _payoutActionState = MutableStateFlow<String?>(null)
    val payoutActionState: StateFlow<String?> = _payoutActionState

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

    // Manual camera-card import. Browse + import are two short sessions, not
    // one long one — holding a session open locks the R6's physical shutter
    // (2026-05-26 Zno-pivot ADR). Both controllers close in finally.
    private val cardBrowseController = UsbCardBrowseController(
        manager = cameraManager.manager,
        deviceProvider = { cameraManager.connectedCameraDevice() },
    )
    private val cardImportController = UsbCardImportController(
        manager = cameraManager.manager,
        deviceProvider = { cameraManager.connectedCameraDevice() },
    )
    private val _cardBrowseState = MutableStateFlow<CardBrowseState>(CardBrowseState.Idle)
    val cardBrowseState: StateFlow<CardBrowseState> = _cardBrowseState
    private var browseJob: Job? = null

    // Increment 4 F — runs alongside an import; if the camera goes Disconnected
    // while we're Importing, cancels the import job and surfaces a friendly
    // Error before the next getObject() would have thrown a cryptic IO error.
    private var disconnectWatcherJob: Job? = null

    private val _shutterWatchState = coordinator.watch
    val shutterWatchState: StateFlow<ShutterWatchState> = _shutterWatchState

    // Account-bound resume intent; individual pending objects live in Room.
    private val tetherPrefs by lazy { TetherWatchPrefs(getApplication<Application>()) }

    init {
        fetchEvents()
        observeQueue()
        fetchEarningsAndTransactions()
        fetchVerificationStatus()
        observeInboxPush()
        cameraManager.start()
        observeNotificationStopRequests()
        observeWatchResume()
        viewModelScope.launch(Dispatchers.IO) {
            try {
                sessionManager.getUserEmail()?.let { coordinator.store.recover(it) }
                runSyncEngine()
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                _cardBrowseState.value = CardBrowseState.Error(e.message ?: "Pending camera photos need recovery. Free phone storage and retry.")
            }
        }
    }

    /** Restore the event snapshot offline; a running service keeps its original destination. */
    private fun observeWatchResume() {
        val eventId = tetherPrefs.activeEventId ?: return
        val owner = sessionManager.getUserEmail() ?: return
        if (tetherPrefs.owner != owner) return
        val saved = runCatching {
            com.google.gson.Gson().fromJson(tetherPrefs.eventSnapshot, PhotographerEventSummaryDto::class.java)
        }.getOrNull()?.takeIf { it.id == eventId }
        if (saved != null) _activeEvent.value = saved
        viewModelScope.launch {
            val ready = combine(_eventsState, cameraManager.state) { events, cam ->
                (saved ?: (events as? EventsState.Success)?.events?.firstOrNull { it.id == eventId })
                    ?.takeIf { cam is CameraConnectionState.Connected }
            }.first { it != null } ?: return@launch
            if (tetherPrefs.activeEventId != eventId ||
                tetherPrefs.owner != sessionManager.getUserEmail()) return@launch
            _activeEvent.value = ready
            if (_shutterWatchState.value is ShutterWatchState.Idle) startShutterWatch()
        }
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

    /** The service handles live Stop itself; close a manual import when the shade requests Stop. */
    private fun observeNotificationStopRequests() {
        viewModelScope.launch {
            TetherEvents.stopRequested.collect {
                val watch = _shutterWatchState.value
                if (watch is ShutterWatchState.Starting || watch is ShutterWatchState.Watching) {
                    stopShutterWatch()
                } else {
                    closeCardImport()
                }
            }
        }
    }

    override fun onCleared() {
        inboxSocket.release()
        super.onCleared()
    }

    fun refreshCameraConnection() {
        cameraManager.refresh()
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
                // Independent endpoints — fired concurrently. Sequential
                // awaits made the Earnings tab pay the sum of three
                // round-trips instead of the slowest one.
                val (overviewResponse, balanceResponse, transactionsResponse) =
                    coroutineScope {
                        val overview = async {
                            RetrofitClient.apiService.getEarningsOverview("Bearer $token")
                        }
                        val balance = async {
                            RetrofitClient.apiService.getPayoutBalance("Bearer $token")
                        }
                        val transactions = async {
                            RetrofitClient.apiService.getTransactionsLedger("Bearer $token")
                        }
                        Triple(overview.await(), balance.await(), transactions.await())
                    }

                if (overviewResponse.success && overviewResponse.data != null &&
                    balanceResponse.success && balanceResponse.data != null &&
                    transactionsResponse.success && transactionsResponse.data != null
                ) {
                    _earningsUiState.value = EarningsUiState.Success(
                        overview = overviewResponse.data,
                        balance = balanceResponse.data,
                        transactions = transactionsResponse.data.items,
                        monthTotals = transactionsResponse.data.monthTotals,
                    )
                } else {
                    val errMsg = overviewResponse.error 
                        ?: balanceResponse.error 
                        ?: transactionsResponse.error 
                        ?: "Failed to load financial records."
                    _earningsUiState.value = EarningsUiState.Error(errMsg)
                }
            } catch (e: Exception) {
                _earningsUiState.value = EarningsUiState.Error(RetrofitClient.parseError(e))
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
                // PAYOUT_NO_ACCOUNT / PAYOUT_BELOW_MINIMUM / PAYOUT_REQUEST_OPEN
                // arrive as non-2xx — surface the server's reason, not
                // "HTTP 409 Conflict".
                val err = RetrofitClient.parseHttpError(e)
                _payoutActionState.value =
                    "Error: ${err?.message ?: RetrofitClient.parseError(e)}"
            }
        }
    }

    fun clearPayoutActionState() {
        _payoutActionState.value = null
    }

    /** Withdraws a HELD payout request (web WithdrawPayoutModal parity). */
    fun withdrawPayoutRequest(cycleId: String) {
        viewModelScope.launch {
            _payoutActionState.value = "processing"
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _payoutActionState.value = "Error: No active login session."
                return@launch
            }
            try {
                val response = RetrofitClient.apiService.withdrawPayout("Bearer $token", cycleId)
                if (response.success) {
                    _payoutActionState.value =
                        "Success: Request withdrawn — you can file a new one when ready."
                    fetchEarningsAndTransactions()
                } else {
                    _payoutActionState.value =
                        "Error: ${response.error ?: "Couldn't withdraw the request."}"
                }
            } catch (e: Exception) {
                val err = RetrofitClient.parseHttpError(e)
                _payoutActionState.value =
                    "Error: ${err?.message ?: RetrofitClient.parseError(e)}"
            }
        }
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
        // IO: buildEventParts reads the cover image bytes off the main thread,
        // matching the other upload paths in this VM.
        viewModelScope.launch(Dispatchers.IO) {
            _eventFormState.value = EventFormState.Submitting
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _eventFormState.value = EventFormState.Error("No valid session. Please log in again.")
                return@launch
            }
            try {
                val parts = buildEventParts(input)
                val response = if (eventId == null) {
                    RetrofitClient.apiService.createPhotographerEvent("Bearer $token", parts)
                } else {
                    RetrofitClient.apiService.updatePhotographerEvent("Bearer $token", eventId, parts)
                }
                if (response.success && response.data != null) {
                    _eventFormState.value = EventFormState.Success(response.data)
                    fetchEvents()
                } else {
                    _eventFormState.value = EventFormState.Error(response.error ?: "Couldn't save the event.")
                }
            } catch (e: Exception) {
                _eventFormState.value = EventFormState.Error(RetrofitClient.parseError(e))
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
            try {
                val response = RetrofitClient.apiService.upsertEventCoupon(
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
                if (response.success && response.data != null) {
                    _couponState.value = CouponUiState.Loaded(response.data)
                } else {
                    _couponActionMessage.value = response.error ?: "Couldn't save the coupon."
                }
            } catch (e: Exception) {
                _couponActionMessage.value = RetrofitClient.parseError(e)
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

    fun selectEvent(event: PhotographerEventSummaryDto?) {
        if (event?.id != _activeEvent.value?.id &&
            (coordinator.isWatching || tetherPrefs.hasActiveEvent)) {
            stopShutterWatch()
        }
        _activeEvent.value = event
    }

    fun queuePhotosFromGallery(context: Context, uris: List<Uri>) {
        val event = _activeEvent.value ?: return
        viewModelScope.launch(Dispatchers.IO) {
            uris.forEachIndexed { index, uri ->
                try {
                    val cacheDir = context.cacheDir
                    val targetFile = File(cacheDir, "gallery_upload_${System.currentTimeMillis()}_$index.jpg")
                    
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        targetFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }

                    if (targetFile.exists() && targetFile.length() > 0) {
                        database.uploadQueueDao().insertRecord(
                            UploadRecord(
                                filePath = targetFile.absolutePath,
                                eventId = event.id,
                                photographerId = sessionManager.getUserEmail() ?: "gallery_upload",
                                captureTimestamp = System.currentTimeMillis(),
                                uploadStatus = "QUEUED"
                            )
                        )
                    }
                } catch (e: Exception) {
                    // Fail silently or handle error log
                }
            }
            
            withContext(Dispatchers.Main) {
                runSyncEngine()
            }
        }
    }


    // Batch-progress bookkeeping for observeQueue — see the comment there.
    private var uploadBatchBaseline = 0
    private var uploadBatchActive = false

    private fun observeQueue() {
        viewModelScope.launch {
            val dao = database.uploadQueueDao()
            combine(
                dao.getStatusCounts(sessionManager.getUserEmail().orEmpty()),
                dao.getRecentRecords(RECENT_RECORDS_LIMIT, sessionManager.getUserEmail().orEmpty()),
                dao.getLatestFailedMessage(sessionManager.getUserEmail().orEmpty()),
                dao.countFailedDuplicates(sessionManager.getUserEmail().orEmpty()),
            ) { counts, recent, latestError, duplicates -> Triple(counts, recent, latestError) to duplicates }
                .collectLatest { (inputs, duplicates) ->
                val (counts, recent, latestError) = inputs
                if (counts.isEmpty()) {
                    _queueStats.value = QueueStats()
                    return@collectLatest
                }

                val stats = queueProgress(counts, uploadBatchBaseline, uploadBatchActive)
                uploadBatchBaseline = stats.baseline
                uploadBatchActive = stats.active

                _queueStats.value = QueueStats(
                    syncedCount = stats.synced,
                    queuedCount = stats.queued,
                    uploadingCount = stats.uploading,
                    failedCount = stats.failed,
                    duplicateCount = duplicates,
                    totalCount = stats.total,
                    progress = stats.progress,
                    lastError = latestError?.takeIf { it.isNotBlank() }?.let(::friendlyUploadError),
                    recentRecords = recent, // already newest-first
                )
            }
        }
    }

    fun runSyncEngine() = PhotoUploadScheduler.enqueue(getApplication())

    /**
     * Flips every FAILED row back to QUEUED with a fresh retry budget and kicks
     * the engine. Terminal-coded rows (duplicate-in-another-event, missing
     * file) settle FAILED again after one attempt with the same message —
     * cheap and honest; same-event duplicates succeed idempotently.
     */
    fun retryFailedUploads() {
        viewModelScope.launch {
            database.uploadQueueDao().requeueFailed(sessionManager.getUserEmail().orEmpty())
            runSyncEngine()
        }
    }

    /** Takes "already in another event" failures off the Failed list; nothing is deleted. */
    fun removeDuplicateUploads() {
        viewModelScope.launch {
            database.uploadQueueDao().markFailedDuplicates(sessionManager.getUserEmail().orEmpty())
        }
    }

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

    /**
     * Start a one-shot enumeration of the JPEGs on the tethered camera's card.
     * Sheet opens immediately (Opening), the controller streams Scanning ticks,
     * then settles into Loaded or Error. Re-entrancy is a no-op while a run is
     * already going so the photographer can't double-tap into two sessions.
     */
    private suspend fun runCameraCard(block: suspend () -> Unit) {
        try { coordinator.runCard {
            sessionManager.getUserEmail()?.let { coordinator.store.recover(it) }
            block()
        } } catch (e: CancellationException) { throw e }
        catch (e: Exception) { _cardBrowseState.value = CardBrowseState.Error(e.message ?: "Could not start the camera service. Allow USB access and retry.") }
    }

    fun browseCameraCard() {
        // The live watch holds the USB interface — a second PtpSession would
        // force-claim it out from under the running session (belt to the UI's
        // disabled Import CTA).
        if (coordinator.isWatching || coordinator.cardBusy.value) return
        if (_cardBrowseState.value !is CardBrowseState.Idle &&
            _cardBrowseState.value !is CardBrowseState.Error
        ) return
        _cardBrowseState.value = CardBrowseState.Opening
        // Capture eventId for the IO leg so the Room lookup is bounded — cards
        // reused across races mustn't bleed dedupe across events.
        val activeEventId = _activeEvent.value?.id
        browseJob?.cancel()
        browseJob = viewModelScope.launch(Dispatchers.IO) {
            val importedSources = if (activeEventId == null) emptySet() else database.uploadQueueDao()
                .getActiveOrCompletedForEvent(activeEventId, sessionManager.getUserEmail().orEmpty())
                .mapNotNullTo(HashSet()) { it.sourceKey }
            runCameraCard {
                cardBrowseController.browse { progress ->
                    kotlinx.coroutines.currentCoroutineContext().ensureActive()
                    when (progress) {
                        is UsbCardBrowseController.Progress.Opening ->
                            _cardBrowseState.value = CardBrowseState.Opening
                        is UsbCardBrowseController.Progress.Scanning ->
                            _cardBrowseState.value = CardBrowseState.Scanning(progress.seen, progress.total)
                        is UsbCardBrowseController.Progress.Done -> {
                            val imported = progress.photos.filter { it.sourceKey in importedSources }.mapTo(HashSet()) { it.handle }
                            val preselected = emptySet<Long>()
                            _cardBrowseState.value = CardBrowseState.Loaded(
                                photos = progress.photos,
                                selectedHandles = preselected,
                                importedHandles = imported,
                            )
                        }
                        is UsbCardBrowseController.Progress.Thumbnails -> {
                            // Only refresh the list the photographer is still looking
                            // at; if they've moved on to importing, the bytes are moot.
                            val current = _cardBrowseState.value as? CardBrowseState.Loaded
                            if (current != null) {
                                _cardBrowseState.value = current.copy(
                                    photos = progress.photos,
                                    thumbsVersion = current.thumbsVersion + 1,
                                )
                            }
                        }
                        is UsbCardBrowseController.Progress.Failed ->
                            _cardBrowseState.value = CardBrowseState.Error(progress.message)
                    }
                }
            }
        }
    }

    /**
     * Dismiss the import sheet and cancel any in-flight scan or transfer.
     * If the user bailed mid-import, the photos already pulled are sitting in
     * Room — kick the sync engine so they actually upload instead of stalling.
     */
    fun closeCardImport() {
        val wasMidImport = _cardBrowseState.value is CardBrowseState.Importing
        browseJob?.cancel()
        browseJob = null
        disconnectWatcherJob?.cancel()
        disconnectWatcherJob = null
        _cardBrowseState.value = CardBrowseState.Idle
        if (wasMidImport) runSyncEngine()
    }

    /**
     * Increment 2 — selection. Toggle / select-all / clear are no-ops unless the
     * sheet is in [CardBrowseState.Loaded]; selection lives on the sealed Loaded
     * state so it resets automatically whenever the photographer re-opens browse.
     */
    fun toggleCardPhotoSelection(handle: Long) {
        val current = _cardBrowseState.value as? CardBrowseState.Loaded ?: return
        val next = if (handle in current.selectedHandles)
            current.selectedHandles - handle
        else
            current.selectedHandles + handle
        _cardBrowseState.value = current.copy(selectedHandles = next)
    }

    fun selectAllCardPhotos() {
        val current = _cardBrowseState.value as? CardBrowseState.Loaded ?: return
        // Skip already-imported (Increment 4 D) — they can't be re-selected.
        _cardBrowseState.value = current.copy(
            selectedHandles = current.photos
                .asSequence()
                .map { it.handle }
                .filter { it !in current.importedHandles }
                .toHashSet()
        )
    }

    fun clearCardPhotoSelection() {
        val current = _cardBrowseState.value as? CardBrowseState.Loaded ?: return
        _cardBrowseState.value = current.copy(selectedHandles = emptySet())
    }

    /** Persist the selected batch, stream each JPEG, and schedule every durable upload. */
    fun importSelectedCardPhotos() {
        val loaded = _cardBrowseState.value as? CardBrowseState.Loaded ?: return
        val selected = loaded.photos.filter {
            it.handle in loaded.selectedHandles && it.handle !in loaded.importedHandles
        }
        runImport(allPhotos = loaded.photos, toImport = selected)
    }

    /**
     * Increment 4 B — re-run just the handles that failed last time. Triggered
     * from the ImportDone sheet view; same code path as the initial import.
     */
    fun retryFailedCardImports() {
        val done = _cardBrowseState.value as? CardBrowseState.ImportDone ?: return
        val toRetry = done.photos.filter { it.handle in done.failedHandles }
        if (toRetry.isEmpty()) return
        runImport(allPhotos = done.photos, toImport = toRetry)
    }

    /** Import a selected snapshot after the previous browse has released its USB session. */
    private fun runImport(allPhotos: List<CardPhoto>, toImport: List<CardPhoto>) {
        val activeEvent = _activeEvent.value ?: return
        if (toImport.isEmpty()) return

        val photographerId = sessionManager.getUserEmail() ?: return

        // The browse may still be streaming thumbnails over the same USB
        // interface; it must have released the session before import claims it.
        val priorBrowse = browseJob
        browseJob?.cancel()
        disconnectWatcherJob?.cancel()

        // Camera-disconnect watcher (Increment 4 F). Aborts the import the
        // moment the cable comes loose so the user sees the cause instead of
        // a cryptic IO error two getObject()s later.
        disconnectWatcherJob = viewModelScope.launch {
            cameraManager.state.collect { camState ->
                if (camState is CameraConnectionState.Disconnected &&
                    _cardBrowseState.value is CardBrowseState.Importing
                ) {
                    browseJob?.cancel()
                    _cardBrowseState.value = CardBrowseState.Error(
                        "Camera disconnected. Re-plug the cable and browse the card again."
                    )
                    runSyncEngine() // flush whatever already queued
                }
            }
        }

        browseJob = viewModelScope.launch(Dispatchers.IO) {
            val watcher = disconnectWatcherJob
            try {
                priorBrowse?.join()
                runCameraCard {
                    coordinator.store.selectForImport(toImport.map { photo ->
                        coordinator.store.transfer(photographerId, activeEvent.id, photo.cameraKey, photo.handle,
                            requireNotNull(photo.objectInfo) { "Browse the camera card again." })
                    })
                    cardImportController.import(
                        photos = toImport,
                        emit = { progress ->
                            kotlinx.coroutines.currentCoroutineContext().ensureActive()
                            _cardBrowseState.value = when (progress) {
                                is UsbCardImportController.Progress.Started ->
                                    CardBrowseState.Importing(
                                        seen = 0,
                                        total = progress.total,
                                        succeeded = 0,
                                        failed = 0,
                                    )
                                is UsbCardImportController.Progress.Each ->
                                    CardBrowseState.Importing(
                                        seen = progress.seen,
                                        total = progress.total,
                                        succeeded = progress.succeededHandles.size,
                                        failed = progress.failedHandles.size,
                                    )
                                is UsbCardImportController.Progress.Done -> {
                                    CardBrowseState.ImportDone(
                                        photos = allPhotos,
                                        succeededHandles = progress.succeededHandles,
                                        failedHandles = progress.failedHandles,
                                    )
                                }
                                is UsbCardImportController.Progress.Failed ->
                                    CardBrowseState.Error(progress.message)
                            }
                        },
                        onFailed = { photo, message ->
                            coordinator.store.fail(coordinator.store.transfer(photographerId, activeEvent.id,
                                photo.cameraKey, photo.handle, requireNotNull(photo.objectInfo)), message)
                        },
                        onPulled = { photo, writeTo ->
                            persistCapturedJpeg(
                                photo,
                                activeEvent.id,
                                photographerId,
                                writeTo,
                            )
                        },
                    )
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                _cardBrowseState.value = CardBrowseState.Error(if (UploadSpool.isFull(e))
                    "Phone storage is full. Free space, then retry the camera import. Originals remain on the card."
                    else e.message ?: "Camera import stopped. Pending photos are kept for retry.")
            } finally {
                watcher?.cancel()
                if (disconnectWatcherJob === watcher) disconnectWatcherJob = null
            }

            // Kick the upload worker if anything actually queued.
            val finalState = _cardBrowseState.value
            if (finalState is CardBrowseState.ImportDone && finalState.succeededHandles.isNotEmpty()) {
                withContext(Dispatchers.Main) { runSyncEngine() }
            }
        }
    }

    /** Persist through the same validated transfer store used by the live service. */
    private suspend fun persistCapturedJpeg(
        photo: CardPhoto,
        eventId: String,
        owner: String,
        writeTo: (OutputStream) -> Unit,
    ): Boolean {
        val info = requireNotNull(photo.objectInfo) { "Browse the camera card again." }
        check(sessionManager.getUserEmail() == owner) { "Sign in to the account that selected these photos." }
        val record = coordinator.store.transfer(owner, eventId, photo.cameraKey, photo.handle, info)
        coordinator.store.save(record, writeTo)
        runSyncEngine()
        return true
    }

    fun startShutterWatch() {
        if (coordinator.isWatching || coordinator.cardBusy.value || browseJob?.isCompleted == false) return
        if (_cardBrowseState.value !is CardBrowseState.Idle && _cardBrowseState.value !is CardBrowseState.Error) return
        if (cameraConnectionState.value !is CameraConnectionState.Connected) return
        val event = _activeEvent.value ?: return
        val owner = sessionManager.getUserEmail() ?: return
        if (!canUploadToEvent(event)) {
            _shutterWatchState.value = ShutterWatchState.Error("This event's upload window has closed.")
            return
        }
        tetherPrefs.saveSnapshot(com.google.gson.Gson().toJson(event))
        coordinator.requestWatch(owner, event.id)
    }

    fun stopShutterWatch() {
        coordinator.stopWatch()
        runSyncEngine()
    }

    fun simulatePhotoCapture() {
        val event = _activeEvent.value ?: return
        viewModelScope.launch(Dispatchers.IO) {
            try {
                // 1. Render a real, decodable JPEG on phone cache storage. A bare
                //    JPEG header is not a valid image — the backend rejects it with
                //    a 500 when it tries to read dimensions / build a thumbnail.
                val cacheDir = getApplication<Application>().cacheDir
                val mockFile = File(cacheDir, "simulated_dslr_${System.currentTimeMillis()}.jpg")

                val width = 1280
                val height = 854
                val centerX = width / 2f
                val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                Canvas(bitmap).apply {
                    // Light pastel background + centered text, so the gallery's
                    // square crop never clips it (a 3:2 frame loses its left/right
                    // edges when displayed in a square tile).
                    drawColor(Color.rgb((205..245).random(), (205..245).random(), (205..245).random()))
                    val title = Paint().apply {
                        color = Color.rgb(60, 60, 60)
                        textSize = 60f
                        isAntiAlias = true
                        textAlign = Paint.Align.CENTER
                        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
                    }
                    val subtitle = Paint().apply {
                        color = Color.rgb(70, 70, 70)
                        textSize = 50f
                        isAntiAlias = true
                        textAlign = Paint.Align.CENTER
                        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
                    }
                    val caption = Paint().apply {
                        color = Color.rgb(120, 120, 120)
                        textSize = 40f
                        isAntiAlias = true
                        textAlign = Paint.Align.CENTER
                    }
                    val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
                        .format(Date())
                    drawText("SIMULATED DSLR FRAME", centerX, 400f, title)
                    drawText(event.name, centerX, 470f, subtitle)
                    drawText(timestamp, centerX, 530f, caption)
                }
                mockFile.outputStream().use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                }
                bitmap.recycle()

                // 2. Insert as a "QUEUED" record in local SQLite database
                database.uploadQueueDao().insertRecord(
                    UploadRecord(
                        filePath = mockFile.absolutePath,
                        eventId = event.id,
                        // Real owner, not a literal — a hardcoded id polluted the
                        // queue's analytics. Matches persistCapturedJpeg().
                        photographerId = sessionManager.getUserEmail() ?: "unknown",
                        captureTimestamp = System.currentTimeMillis(),
                        uploadStatus = "QUEUED"
                    )
                )
                // Kick the worker like every other enqueue path (gallery, card
                // import, live capture) — WorkManager is thread-safe from IO.
                runSyncEngine()
            } catch (e: Exception) {
                // Fail silently or log error during simulation
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

/**
 * One tick of the upload queue's counters, plus the batch bookkeeping the next
 * tick needs back. [baseline] and [active] are the caller's carried state.
 */
internal data class QueueProgress(
    val synced: Int,
    val queued: Int,
    val uploading: Int,
    val failed: Int,
    val total: Int,
    val progress: Float,
    /** Rows already settled when the current batch began — feed this back as `previousBaseline`. */
    val baseline: Int,
    val active: Boolean,
)

/**
 * Progress is BATCH-relative. COMPLETED rows persist forever as the card-import
 * dedupe ledger, so `synced/total` over all history meant a 500-photo race
 * followed by one new queued photo showed 500/501 ≈ 99% instantly. The baseline
 * is the rows already settled when the current batch began (the idle → active
 * edge); it only moves on that edge, so a batch's own completions fill the bar.
 * DUPLICATE counts as settled, like FAILED, so the bar still reaches 100%.
 */
internal fun queueProgress(
    counts: List<StatusCount>,
    previousBaseline: Int,
    wasActive: Boolean,
): QueueProgress {
    val byStatus = counts.associate { it.status to it.count }
    val synced = byStatus["COMPLETED"] ?: 0
    val queued = byStatus["QUEUED"] ?: 0
    val uploading = byStatus["UPLOADING"] ?: 0
    val failed = byStatus["FAILED"] ?: 0
    val duplicate = byStatus["DUPLICATE"] ?: 0
    val total = counts.sumOf { it.count }
    val settled = synced + failed + duplicate

    val active = queued + uploading > 0
    val baseline = if (active && !wasActive) settled else previousBaseline
    val batchTotal = total - baseline
    return QueueProgress(
        synced = synced,
        queued = queued,
        uploading = uploading,
        failed = failed,
        total = total,
        progress = when {
            !active -> if (total > 0) 1f else 0f
            batchTotal > 0 -> (settled - baseline).toFloat() / batchTotal.toFloat()
            else -> 0f
        },
        baseline = baseline,
        active = active,
    )
}
