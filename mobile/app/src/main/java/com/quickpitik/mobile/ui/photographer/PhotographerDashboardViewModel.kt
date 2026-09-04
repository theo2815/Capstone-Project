package com.quickpitik.mobile.ui.photographer

import android.app.Application
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.net.Uri
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.quickpitik.mobile.BuildConfig
import com.quickpitik.mobile.data.MAX_UPLOAD_BYTES
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.TetherEvents
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.data.local.UploadSpool
import java.util.concurrent.TimeUnit
import com.quickpitik.mobile.data.readAtMost
import com.quickpitik.mobile.data.remote.EarningsOverviewDto
import com.quickpitik.mobile.data.remote.PayoutBalanceDto
import com.quickpitik.mobile.data.remote.PhotoExistsRequest
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.data.remote.PhotographerMessageFrame
import com.quickpitik.mobile.data.remote.PhotographerTransactionDto
import com.quickpitik.mobile.data.remote.QpWebSocket
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.WsState
import com.quickpitik.mobile.data.usb.CameraConnectionManager
import com.quickpitik.mobile.data.usb.CameraConnectionState
import com.quickpitik.mobile.data.usb.ptp.CardPhoto
import com.quickpitik.mobile.data.usb.ptp.UsbCardBrowseController
import com.quickpitik.mobile.data.usb.ptp.UsbCardImportController
import com.quickpitik.mobile.data.usb.ptp.UsbEventCaptureController
import com.quickpitik.mobile.service.TetherIngestService
import com.quickpitik.mobile.ui.runner.canUploadToEvent
import com.quickpitik.mobile.worker.PhotoUploadWorker
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
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

sealed class SharePhotosState {
    object Loading : SharePhotosState()
    data class Success(val photos: List<com.quickpitik.mobile.data.remote.PhotographerLibraryPhotoDto>) : SharePhotosState()
    data class Error(val message: String) : SharePhotosState()
}

data class QueueStats(
    val syncedCount: Int = 0,
    val queuedCount: Int = 0,
    val uploadingCount: Int = 0,
    val failedCount: Int = 0,
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

/**
 * UI-side state for live auto-upload ("shutter watch"): the photographer shoots
 * on the tethered camera and every JPEG flows into the upload queue. `Idle`
 * shows the start CTA; `Watching` is the live session with its rolling log.
 */
sealed class ShutterWatchState {
    object Idle : ShutterWatchState()
    object Starting : ShutterWatchState()
    data class Watching(
        val captureCount: Int,
        val lastCaptureName: String?,
        /** Rolling tail of controller log lines (max [WATCH_LOG_LINES], newest last). */
        val recentLog: List<String>,
        /**
         * The camera link dropped and the controller is reopening it. Still
         * [Watching] rather than [Error] on purpose: the shoot is paused, not
         * over, and the state is what keeps the foreground service (and its
         * wakelock) alive while we wait for the camera to come back.
         */
        val reconnecting: Boolean = false,
    ) : ShutterWatchState()
    /** [recentLog] = the controller's last lines, so a release build (no logcat) still shows WHY. */
    data class Error(val message: String, val recentLog: List<String> = emptyList()) : ShutterWatchState()
}

private const val WATCH_LOG_LINES = 12
// Newest rows the Capture tab observes for its strip (the screen shows ≤30).
private const val RECENT_RECORDS_LIMIT = 200
// Prefs key: event id of a live shutter watch, so a relaunch after process
// death resumes it instead of waiting for a tap nobody knows is needed.
private const val WATCH_ACTIVE_EVENT_KEY = "watch_active_event"

class PhotographerDashboardViewModel(application: Application) : AndroidViewModel(application) {
    private val database = AppDatabase.getDatabase(application)
    private val sessionManager = SessionManager.getInstance(application)
    private val workManager = WorkManager.getInstance(application)
    private val cameraManager = CameraConnectionManager(application)
    val cameraConnectionState: StateFlow<CameraConnectionState> = cameraManager.state

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

    private val _isFetchingBrandSettings = MutableStateFlow(false)
    val isFetchingBrandSettings: StateFlow<Boolean> = _isFetchingBrandSettings

    private val _brandSettings = MutableStateFlow<com.quickpitik.mobile.data.remote.BrandSettingsResponseDto?>(null)
    val brandSettings: StateFlow<com.quickpitik.mobile.data.remote.BrandSettingsResponseDto?> = _brandSettings

    // Non-null when the settings hydration failed with nothing cached — the
    // settings screen must not offer an empty editable form (see fetchSettings).
    private val _settingsLoadError = MutableStateFlow<String?>(null)
    val settingsLoadError: StateFlow<String?> = _settingsLoadError

    // Backend-owned PH region/province list (GET /regions). Empty until the
    // fetch lands; the region slab shows "Loading regions…" meanwhile rather
    // than an empty picker. A failed fetch leaves it empty and the photographer
    // keeps whatever region is already saved — nothing is silently cleared.
    private val _regions = MutableStateFlow<List<com.quickpitik.mobile.data.remote.RegionDto>>(emptyList())
    val regions: StateFlow<List<com.quickpitik.mobile.data.remote.RegionDto>> = _regions

    private val _payoutAccounts = MutableStateFlow<List<com.quickpitik.mobile.data.remote.PayoutAccountDto>>(emptyList())
    val payoutAccounts: StateFlow<List<com.quickpitik.mobile.data.remote.PayoutAccountDto>> = _payoutAccounts

    private val _socials = MutableStateFlow<List<com.quickpitik.mobile.data.remote.SocialLinkDto>>(emptyList())
    val socials: StateFlow<List<com.quickpitik.mobile.data.remote.SocialLinkDto>> = _socials

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

    // Increment 4 D — handles that succeeded in any import during this VM
    // lifetime. Re-browsing the same card shows them as "Imported" so the
    // photographer can't accidentally re-pull. Reset when the camera detaches
    // (handle space is per-storage; a different body has its own counter).
    private var sessionImportedHandles: Set<Long> = emptySet()

    // Increment 4 F — runs alongside an import; if the camera goes Disconnected
    // while we're Importing, cancels the import job and surfaces a friendly
    // Error before the next getObject() would have thrown a cryptic IO error.
    private var disconnectWatcherJob: Job? = null

    // Live auto-upload ("shutter watch"). Unlike browse/import, this HOLDS a
    // PTP session in EOS event mode for the whole shooting session — which is
    // exactly why it must never run at the same time as the card flow (both
    // force-claim the same USB interface).
    private val shutterWatchController = UsbEventCaptureController(
        manager = cameraManager.manager,
        deviceProvider = { cameraManager.connectedCameraDevice() },
    )
    private val _shutterWatchState = MutableStateFlow<ShutterWatchState>(ShutterWatchState.Idle)
    val shutterWatchState: StateFlow<ShutterWatchState> = _shutterWatchState
    private var shutterWatchJob: Job? = null
    private var shutterWatchDetachJob: Job? = null

    // Rolling log tail behind ShutterWatchState. Only mutated from the
    // controller's single coroutine (onLog/onStarted are sequential there),
    // so a plain var is safe.
    private var watchLogTail: List<String> = emptyList()

    // One low-storage warning per watch session (reset in startShutterWatch).
    @Volatile
    private var lowStorageWarned = false

    // Highest camera object handle successfully handled per event. Lets the
    // next Start catch up on frames shot while auto-upload was stopped instead
    // of silently re-baselining them away (14 frames lost that way 2026-09-02).
    private val tetherPrefs by lazy {
        getApplication<Application>().getSharedPreferences("quickpitik_tether", Context.MODE_PRIVATE)
    }

    init {
        fetchEvents()
        observeQueue()
        fetchEarningsAndTransactions()
        fetchVerificationStatus()
        fetchBrandSettings()
        observeInboxPush()
        cameraManager.start()
        observeCameraDetach()
        observeIngestForService()
        observeNotificationStopRequests()
        observeWatchResume()
    }

    /**
     * Auto-resume after process death. If the app was killed mid-shoot (OOM,
     * a crash, the user swiping it away), the prefs flag written by
     * [startShutterWatch] survives while the coroutine did not. The first time
     * this VM sees the events loaded AND a camera connected, it re-selects that
     * event and restarts the watch — the catch-up logic then pulls whatever was
     * shot in between. One-shot; a deliberate Stop or a normal logout clears
     * the flag so this never surprises anyone.
     */
    private fun observeWatchResume() {
        val eventId = tetherPrefs.getString(WATCH_ACTIVE_EVENT_KEY, null) ?: return
        viewModelScope.launch {
            val ready = combine(_eventsState, cameraManager.state) { events, cam ->
                (events as? EventsState.Success)?.events?.firstOrNull { it.id == eventId }
                    ?.takeIf { cam is CameraConnectionState.Connected }
            }.first { it != null } ?: return@launch
            if (_shutterWatchState.value !is ShutterWatchState.Idle) return@launch
            if (_activeEvent.value?.id != ready.id) selectEvent(ready)
            appendWatchLog("Resuming auto-upload for ${ready.name} after the app was interrupted.")
            startShutterWatch()
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

    /**
     * Keeps [TetherIngestService] running for exactly as long as bytes are
     * moving off the camera, and feeds it the shade copy.
     *
     * Driven off the two ingest states rather than sprinkled through the six
     * start / stop / error / detach paths: those paths already write the states
     * read here, so none of them needed an edit, and there is no ref-count to
     * get wrong when the watch and the card flow hand off to each other.
     */
    private fun observeIngestForService() {
        viewModelScope.launch {
            combine(_shutterWatchState, _cardBrowseState) { watch, card ->
                ingestNotice(watch, card)
            }
                .distinctUntilChanged()
                .collect { notice ->
                    if (notice == null) stopIngestService() else startOrUpdateIngestService(notice)
                }
        }
    }

    /**
     * Shade copy for the ingest currently in flight, or null when none is.
     *
     * [CardBrowseState.Loaded] and [CardBrowseState.ImportDone] are deliberately
     * null: the sheet is open but no USB I/O is running — the photographer is
     * choosing photos. Holding a foreground notification while someone reads a
     * list is user-hostile, and it's the pattern Play policy flags.
     */
    private fun ingestNotice(watch: ShutterWatchState, card: CardBrowseState): String? = when {
        watch is ShutterWatchState.Starting -> "Starting auto-upload…"
        // Deliberately still a notice, so the service and its wakelock stay up
        // across the outage. Letting the process go cached here is exactly what
        // would stop us ever noticing the camera came back.
        watch is ShutterWatchState.Watching && watch.reconnecting ->
            "Auto-upload paused · reconnecting to camera…"
        watch is ShutterWatchState.Watching -> {
            val n = watch.captureCount
            "Auto-upload live · $n photo${if (n == 1) "" else "s"} sent"
        }
        card is CardBrowseState.Opening -> "Reading camera card…"
        card is CardBrowseState.Scanning -> "Reading camera card · ${card.seen} of ${card.total}"
        card is CardBrowseState.Importing -> "Importing photos · ${card.seen} of ${card.total}"
        else -> null
    }

    private fun startOrUpdateIngestService(notice: String) {
        val context = getApplication<Application>()
        val intent = Intent(context, TetherIngestService::class.java)
            .setAction(TetherIngestService.ACTION_START)
            .putExtra(TetherIngestService.EXTRA_NOTICE, notice)
        // Every first start follows a tap on the Capture tab, so the app is in
        // the foreground and API 31+'s background-start rule is satisfied; later
        // calls target an already-foreground service. runCatching is the
        // backstop — a ForegroundServiceStartNotAllowedException must never take
        // down a live shoot, and losing the lifeline beats crashing out of it.
        runCatching { ContextCompat.startForegroundService(context, intent) }
    }

    private fun stopIngestService() {
        val context = getApplication<Application>()
        context.stopService(Intent(context, TetherIngestService::class.java))
    }

    /**
     * "Stop" in the notification shade. The service holds no reference to the
     * controllers, so it raises a signal and we end whichever flow is live.
     * [closeCardImport] is a no-op when the card flow is already idle.
     */
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

    /**
     * Reset the dedupe set whenever the camera detaches — handles are per-
     * storage so a fresh body gets its own counter starting at 1. Carries
     * Increment 4 D (dedupe) correctness when the photographer swaps bodies.
     */
    private fun observeCameraDetach() {
        viewModelScope.launch {
            cameraManager.state.collect { state ->
                if (state is CameraConnectionState.Disconnected) {
                    sessionImportedHandles = emptySet()
                }
            }
        }
    }

    override fun onCleared() {
        // Logging out mid-ingest tears down this ViewModel and its coroutines;
        // without this the shade would keep a notification claiming a shoot is
        // still running.
        stopIngestService()
        cameraManager.stop()
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
            _eventsState.value = EventsState.Loading
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
                    _eventsState.value = EventsState.Error(response.error ?: "Failed to load events.")
                }
            } catch (e: Exception) {
                _eventsState.value = EventsState.Error(e.localizedMessage ?: "Failed to connect to server.")
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

    fun selectEvent(event: PhotographerEventSummaryDto?) {
        // A live watch is bound to the event it started with — switching away
        // (or backing out to the picker) must stop it, or frames would keep
        // queueing to the old event's snapshot.
        if (event?.id != _activeEvent.value?.id &&
            (_shutterWatchState.value is ShutterWatchState.Starting ||
                _shutterWatchState.value is ShutterWatchState.Watching)
        ) {
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
                dao.getStatusCounts(),
                dao.getRecentRecords(RECENT_RECORDS_LIMIT),
                dao.getLatestFailedMessage(),
            ) { counts, recent, latestError -> Triple(counts, recent, latestError) }
                .collectLatest { (counts, recent, latestError) ->
                if (counts.isEmpty()) {
                    _queueStats.value = QueueStats()
                    return@collectLatest
                }

                val byStatus = counts.associate { it.status to it.count }
                val synced = byStatus["COMPLETED"] ?: 0
                val queued = byStatus["QUEUED"] ?: 0
                val uploading = byStatus["UPLOADING"] ?: 0
                val failed = byStatus["FAILED"] ?: 0
                val total = counts.sumOf { it.count }

                // Progress is BATCH-relative. COMPLETED rows persist forever as
                // the card-import dedupe ledger, so `synced/total` over all
                // history meant a 500-photo race followed by one new queued
                // photo showed 500/501 ≈ 99% instantly. Baseline = rows already
                // settled when the current batch began (idle → active edge).
                val active = queued + uploading > 0
                if (active && !uploadBatchActive) uploadBatchBaseline = synced + failed
                uploadBatchActive = active
                val batchTotal = total - uploadBatchBaseline
                val batchDone = (synced + failed) - uploadBatchBaseline
                val progress = when {
                    !active -> if (total > 0) 1f else 0f
                    batchTotal > 0 -> batchDone.toFloat() / batchTotal.toFloat()
                    else -> 0f
                }

                _queueStats.value = QueueStats(
                    syncedCount = synced,
                    queuedCount = queued,
                    uploadingCount = uploading,
                    failedCount = failed,
                    totalCount = total,
                    progress = progress,
                    lastError = latestError?.takeIf { it.isNotBlank() },
                    recentRecords = recent, // already newest-first
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
            // Linear 10 s, not the default exponential 30 s → 5 h: during a
            // race a backend blip must cost seconds of drain, not hours.
            .setBackoffCriteria(BackoffPolicy.LINEAR, 10, TimeUnit.SECONDS)
            .build()

        // Unique + KEEP: only one sync worker ever runs, so two workers can't
        // race the same QUEUED snapshot and double-upload. KEEP is enough
        // because the worker drains the queue in a loop — rows inserted while
        // it runs are picked up by its next pass. Residual race (a row landing
        // in the worker's final ms while KEEP swallows the kick) is covered by
        // the live path's per-capture kicks and the visible "Run sync engine"
        // button; APPEND_OR_REPLACE would close it fully if ever needed.
        workManager.enqueueUniqueWork(
            "photo-upload-sync",
            ExistingWorkPolicy.KEEP,
            syncRequest,
        )
    }

    /**
     * Wipes every FAILED row from the sync queue. A HTTP 500 batch on
     * 2026-05-28 left 46 stale rows that the user just wanted gone — root
     * cause (which server-side exception fired) was deferred to a future
     * backend session. Files on disk are intentionally left to the OS cache
     * reaper; we don't own arbitrary filePaths and touching them is risky.
     */
    fun clearFailedUploads() {
        viewModelScope.launch {
            database.uploadQueueDao().deleteByStatus("FAILED")
        }
    }

    /**
     * Flips every FAILED row back to QUEUED with a fresh retry budget and kicks
     * the engine. Terminal-coded rows (duplicate-in-another-event, missing
     * file) settle FAILED again after one attempt with the same message —
     * cheap and honest; same-event duplicates succeed idempotently.
     */
    fun retryFailedUploads() {
        viewModelScope.launch {
            database.uploadQueueDao().requeueFailed()
            runSyncEngine()
        }
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

    fun fetchBrandSettings() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            loadBrandSettings(token)
        }
    }

    private suspend fun loadBrandSettings(token: String) {
        _isFetchingBrandSettings.value = true
        try {
            val response = RetrofitClient.apiService.getBrandSettings("Bearer $token")
            if (response.success && response.data != null) {
                _brandSettings.value = response.data
                _settingsLoadError.value = null
            }
        } catch (e: Exception) {
            // The brand payload hydrates the whole settings form. With nothing
            // cached, an empty editable form could overwrite server values.
            if (_brandSettings.value == null) {
                _settingsLoadError.value = RetrofitClient.parseError(e)
            }
        } finally {
            _isFetchingBrandSettings.value = false
        }
    }

    fun fetchSettings() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            coroutineScope {
                launch { loadBrandSettings(token) }
                launch {
                    try {
                        val response = RetrofitClient.apiService.getPayoutAccounts("Bearer $token")
                        if (response.success && response.data != null) {
                            _payoutAccounts.value = response.data
                        }
                    } catch (_: Exception) {
                        // Keep the last known list.
                    }
                }
                launch {
                    try {
                        val response = RetrofitClient.apiService.getSocials("Bearer $token")
                        if (response.success && response.data != null) {
                            _socials.value = response.data
                        }
                    } catch (_: Exception) {
                        // Keep the last known list.
                    }
                }
                // Reference data changes rarely and is cached by the backend.
                if (_regions.value.isEmpty()) launch {
                    try {
                        val response = RetrofitClient.apiService.getRegions()
                        if (response.success && response.data != null) {
                            _regions.value = response.data
                        }
                    } catch (_: Exception) {
                        // Keep the saved region visible.
                    }
                }
            }
        }
    }

    /**
     * Start a one-shot enumeration of the JPEGs on the tethered camera's card.
     * Sheet opens immediately (Opening), the controller streams Scanning ticks,
     * then settles into Loaded or Error. Re-entrancy is a no-op while a run is
     * already going so the photographer can't double-tap into two sessions.
     */
    fun browseCameraCard() {
        // The live watch holds the USB interface — a second PtpSession would
        // force-claim it out from under the running session (belt to the UI's
        // disabled Import CTA).
        if (_shutterWatchState.value is ShutterWatchState.Starting ||
            _shutterWatchState.value is ShutterWatchState.Watching
        ) return
        if (_cardBrowseState.value !is CardBrowseState.Idle &&
            _cardBrowseState.value !is CardBrowseState.Error
        ) return
        _cardBrowseState.value = CardBrowseState.Opening
        // Capture eventId for the IO leg so the Room lookup is bounded — cards
        // reused across races mustn't bleed dedupe across events.
        val activeEventId = _activeEvent.value?.id
        browseJob?.cancel()
        browseJob = viewModelScope.launch(Dispatchers.IO) {
            // Persistent-dedupe set: original filenames of records already in
            // the local upload queue for THIS event (QUEUED/UPLOADING/COMPLETED).
            // Survives app restarts so re-plugging a card after the worker drained
            // doesn't re-show the same photos as importable. Backend is FROZEN
            // per Build Mandate — no cross-device round-trip here.
            val persistentlyImportedFilenames: Set<String> =
                if (activeEventId.isNullOrBlank()) emptySet()
                else database.uploadQueueDao()
                    .getActiveOrCompletedForEvent(activeEventId)
                    .mapNotNullTo(HashSet()) { extractOriginalCardFilename(it.filePath) }

            // "New since you last shot here": frames above this event's live
            // watch high-water mark are pre-selected, so the common post-race
            // import is one tap. Same handle-family rule as the controller.
            val highWater = activeEventId
                ?.let { tetherPrefs.getLong("hw_$it", -1L) }
                ?.takeIf { it >= 0 }

            cardBrowseController.browse { progress ->
                when (progress) {
                    is UsbCardBrowseController.Progress.Opening ->
                        _cardBrowseState.value = CardBrowseState.Opening
                    is UsbCardBrowseController.Progress.Scanning ->
                        _cardBrowseState.value = CardBrowseState.Scanning(progress.seen, progress.total)
                    is UsbCardBrowseController.Progress.Done -> {
                        // Two paths to "Imported": (a) in-session handle dedupe
                        // (camera handle space, resets on detach) and (b) Room-
                        // persistent filename dedupe (this event, this device).
                        val imported = progress.photos
                            .filter { p ->
                                p.handle in sessionImportedHandles ||
                                    p.filename in persistentlyImportedFilenames
                            }
                            .mapTo(HashSet()) { it.handle }
                        val preselected = if (highWater == null) emptySet() else progress.photos
                            .filter { p ->
                                p.handle !in imported && p.handle > highWater &&
                                    (p.handle ushr 16) == (highWater ushr 16)
                            }
                            .mapTo(HashSet()) { it.handle }
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

    /**
     * Recover the original camera filename (e.g. `R6T_1083.JPG`) from the cache
     * file path the import flow writes: `<cacheDir>/dslr_import_<ts>_<original>`.
     * Returns null for cache files written by other paths (`simulated_dslr_*`,
     * `gallery_upload_*`) so they don't pollute card-dedupe.
     */
    private fun extractOriginalCardFilename(filePath: String): String? {
        val name = File(filePath).name
        if (!name.startsWith("dslr_import_")) return null
        // Format: dslr_import_<timestamp>_<original> — original may itself
        // contain underscores (Canon's R6T_1083.JPG), so split with limit=4
        // and take the 4th piece intact.
        val parts = name.split('_', limit = 4)
        return if (parts.size == 4 && parts[3].isNotBlank()) parts[3] else null
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

    /**
     * Increment 3 — pull the selected handles off the card, write each JPEG to
     * the app cache, enqueue an `UploadRecord`, and kick the existing
     * `PhotoUploadWorker` once everything is queued. Same persistence pattern
     * as `simulatePhotoCapture`, so the existing sync-queue UI surfaces the
     * uploads with no further changes.
     *
     * Per-photo failures don't abort the run; the summary reports the tally.
     * `closeCardImport()` will flush partial imports if the user dismisses
     * mid-flight.
     */
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

    /**
     * Shared import path used by both the initial import and the retry-failed
     * action. [allPhotos] is the full card list — preserved on ImportDone so
     * retry-failed can be re-triggered repeatedly without re-browsing.
     *
     * Side effects: starts the [UsbCardImportController] on `Dispatchers.IO`,
     * runs the [disconnectWatcherJob] in parallel (Increment 4 F), and on Done
     * kicks the [PhotoUploadWorker] when anything queued. Also folds the run's
     * succeeded handles into [sessionImportedHandles] so future browse Loaded
     * states mark them as Imported (Increment 4 D).
     */
    private fun runImport(allPhotos: List<CardPhoto>, toImport: List<CardPhoto>) {
        val activeEvent = _activeEvent.value ?: return
        if (toImport.isEmpty()) return

        val photographerId = sessionManager.getUserEmail() ?: "unknown"

        // recordId → SHA-256, filled as frames land. Consumed once at the end
        // by the dedup pre-flight below.
        val importedHashes = mutableMapOf<Long, String>()

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
            priorBrowse?.join()
            cardImportController.import(
                photos = toImport,
                emit = { progress ->
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
                            sessionImportedHandles = sessionImportedHandles + progress.succeededHandles
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
                onPulled = { photo, writeTo ->
                    persistCapturedJpeg(
                        photo.filename,
                        activeEvent.id,
                        photographerId,
                        writeTo,
                    ) { recordId, hash -> importedHashes[recordId] = hash }
                },
            )
            // Tear down the watcher (whether we finished cleanly or the watcher
            // already cancelled us) so it doesn't outlive the import.
            disconnectWatcherJob?.cancel()
            disconnectWatcherJob = null

            // Drop anything the backend already holds BEFORE the worker starts,
            // so duplicate bytes never go over the wire.
            dropAlreadyUploaded(activeEvent.id, importedHashes)

            // Kick the upload worker if anything actually queued.
            val finalState = _cardBrowseState.value
            if (finalState is CardBrowseState.ImportDone && finalState.succeededHandles.isNotEmpty()) {
                withContext(Dispatchers.Main) { runSyncEngine() }
            }
        }
    }

    /**
     * Dedup pre-flight (backend Phase 2, `POST …/photos/exists`). Asks which of
     * the just-imported hashes the backend already holds and de-queues the ones
     * already stored against THIS event, so their bytes are never uploaded.
     *
     * Mobile already dedupes twice locally — in-session PTP handles, and a
     * Room-persistent per-event filename check. This closes the case neither
     * catches: the same bytes arriving under a different filename, or a card
     * re-imported after the app's data was cleared. The website has run this
     * pre-flight since 2026-06-02; mobile was the only client without it.
     *
     * `different_event` is deliberately left queued. That upload will 409, and
     * `PhotoUploadWorker` already treats `PHOTO_DUPLICATE_DIFFERENT_EVENT` as
     * terminal — the photographer should see it failed rather than have the
     * frame silently vanish, because uploading it to the wrong event is a real
     * mistake worth surfacing.
     *
     * Best-effort: any failure leaves everything queued, which is exactly the
     * behaviour before this existed.
     */
    private suspend fun dropAlreadyUploaded(eventId: String, hashes: Map<Long, String>) {
        if (hashes.isEmpty()) return
        val token = sessionManager.getAccessToken() ?: return
        try {
            // Backend caps the list at 500 per request.
            val skipped = ArrayList<Long>()
            // recordId → name of the event that already holds these bytes.
            val elsewhere = HashMap<Long, String?>()
            hashes.entries.chunked(500).forEach { chunk ->
                val response = RetrofitClient.apiService.checkPhotosExist(
                    "Bearer $token",
                    eventId,
                    PhotoExistsRequest(hashes = chunk.map { it.value }),
                )
                val byHash = response.data?.results.orEmpty().associateBy { it.hash }
                for ((recordId, hash) in chunk) {
                    when (byHash[hash]?.status) {
                        "same_event" -> skipped.add(recordId)
                        "different_event" -> elsewhere[recordId] = byHash[hash]?.eventName
                    }
                }
            }
            val dao = database.uploadQueueDao()
            for (recordId in skipped) {
                // Delete the cached file too — nothing will read it now.
                dao.getRecordById(recordId)?.let { runCatching { File(it.filePath).delete() } }
                dao.deleteRecordById(recordId)
            }
            // The upload would 409 (terminal) anyway — device-verified 2026-09-02:
            // two 1.5 MB frames went up the wire twice each just to be rejected.
            // Settle them FAILED now, with the same reason the backend would give,
            // so the photographer sees WHY without paying for the round trip.
            for ((recordId, eventName) in elsewhere) {
                dao.updateStatus(
                    recordId,
                    "FAILED",
                    "Skipped — already in your event '${eventName ?: "another event"}', so it wasn't uploaded twice.",
                )
            }
            if (BuildConfig.DEBUG) {
                android.util.Log.i(
                    "QP/UPLOAD-PERF",
                    "dedup pre-flight skipped=${skipped.size} elsewhere=${elsewhere.size} of ${hashes.size}",
                )
            }
        } catch (e: Exception) {
            // Offline or a backend hiccup — keep everything queued. The upload
            // path stays correct without this; it is only an optimisation.
            if (BuildConfig.DEBUG) {
                android.util.Log.w("QP/UPLOAD-PERF", "dedup pre-flight failed: ${e.message}")
            }
        }
    }

    /**
     * Write a pulled JPEG to the app cache and enqueue an [UploadRecord] for
     * [PhotoUploadWorker]. Shared by manual card import and live auto-upload.
     * The cache name keeps the `dslr_import_<ts>_<original>` convention so
     * [extractOriginalCardFilename]'s per-event dedupe also recognizes frames
     * that arrived via the live watch — they render as "Imported" in the card
     * browse sheet instead of being offered for a second pull.
     */
    private suspend fun persistCapturedJpeg(
        originalFilename: String,
        eventId: String,
        photographerId: String,
        writeTo: (OutputStream) -> Unit,
        // Receives (queued record id, SHA-256 of the bytes) for each frame that
        // lands. Card import collects these to run one dedup pre-flight for the
        // whole batch; the live shutter path passes null (one frame at a time
        // is not a batch, and it already dedupes on handle).
        hashSink: ((Long, String) -> Unit)? = null,
    ): Boolean {
        val persistStart = System.currentTimeMillis()
        val app = getApplication<Application>()
        // Refuse to pull onto a nearly-full phone: a frame that can't be written
        // whole is worse than one left on the card, where card import can still
        // fetch it once space is freed. Warned once per watch, not per frame.
        if (!UploadSpool.hasRoom(app)) {
            if (!lowStorageWarned) {
                lowStorageWarned = true
                appendWatchLog(
                    "Phone storage is nearly full — pausing pulls from the camera. Free up " +
                        "space, then use Import from camera card to fetch the frames left behind."
                )
            }
            return false
        }
        val unique = "dslr_import_${System.currentTimeMillis()}_$originalFilename"
        // filesDir spool, not cacheDir — the OS may evict cache before upload.
        val file = File(UploadSpool.dir(app), unique)
        return try {
            // The PTP read happens inside here, streaming straight to disk, so
            // a frame never exists whole in memory. The digest rides the same
            // stream, so hashing costs no extra pass over the bytes.
            val digest = java.security.MessageDigest.getInstance("SHA-256")
            file.outputStream().buffered().use { out ->
                writeTo(java.security.DigestOutputStream(out, digest))
            }
            // An empty file means the camera handed over nothing (an object
            // still being written to the card, typically). Leave no orphan and
            // report failure so the caller retries the handle.
            if (file.length() == 0L) {
                file.delete()
                return false
            }
            val recordId = database.uploadQueueDao().insertRecord(
                UploadRecord(
                    filePath = file.absolutePath,
                    eventId = eventId,
                    photographerId = photographerId,
                    captureTimestamp = System.currentTimeMillis(),
                    uploadStatus = "QUEUED",
                )
            )
            hashSink?.invoke(recordId, digest.digest().joinToString("") { "%02x".format(it) })
            val persistMs = System.currentTimeMillis() - persistStart
            if (BuildConfig.DEBUG) {
                android.util.Log.i(
                    "QP/UPLOAD-PERF",
                    "persist file=$originalFilename bytes=${file.length()} ms=$persistMs",
                )
            }
            true
        } catch (e: Exception) {
            // Streaming means a mid-transfer failure leaves a partial file
            // behind; the buffered version could never produce one. Nothing
            // was queued, so the file has no owner — drop it.
            runCatching { file.delete() }
            false
        }
    }

    /**
     * Live auto-upload: hold a PTP session in EOS event mode and pipe every
     * shutter press through [persistCapturedJpeg] → [PhotoUploadWorker]. The
     * event is snapshotted at start; [selectEvent] stops the watch on a real
     * switch so frames can't leak into the wrong event.
     */
    fun startShutterWatch() {
        // Cheap state gates, re-checked here even though the UI disables the CTA.
        val current = _shutterWatchState.value
        if (current is ShutterWatchState.Starting || current is ShutterWatchState.Watching) return
        // The card browse/import flow owns the USB interface while active —
        // two PtpSessions force-claiming the same interface corrupt each other.
        if (_cardBrowseState.value !is CardBrowseState.Idle &&
            _cardBrowseState.value !is CardBrowseState.Error
        ) return
        if (cameraConnectionState.value !is CameraConnectionState.Connected) return
        val event = _activeEvent.value ?: return
        if (!canUploadToEvent(event.date)) {
            _shutterWatchState.value = ShutterWatchState.Error("This event's upload window has closed.")
            return
        }

        val eventId = event.id
        val photographerId = sessionManager.getUserEmail() ?: "unknown"
        watchLogTail = emptyList()
        lowStorageWarned = false
        val highWaterKey = "hw_$eventId"
        // ponytail: keyed by event only, not by camera body/card. A different
        // card with higher handles could trigger a catch-up of up to the
        // controller's cap; the backend's same-event hash dedupe makes that
        // bandwidth, never duplicates. Add a camera key if it ever bites.
        val catchUpAfter = tetherPrefs.getLong(highWaterKey, -1L).takeIf { it >= 0 }
        tetherPrefs.edit().putString(WATCH_ACTIVE_EVENT_KEY, eventId).apply()
        _shutterWatchState.value = ShutterWatchState.Starting

        // Detach watcher. It no longer CANCELS the watch: the controller now
        // reopens a dropped session itself (its deviceProvider already returns
        // null while detached), so a knocked cable is a pause, not the end of
        // the shoot. All this does is flush what's queued and surface the pause
        // immediately, rather than waiting for the controller's own detection.
        shutterWatchDetachJob?.cancel()
        shutterWatchDetachJob = viewModelScope.launch {
            cameraManager.state.collect { camState ->
                if (camState !is CameraConnectionState.Disconnected) return@collect
                _shutterWatchState.update { s ->
                    if (s is ShutterWatchState.Watching) s.copy(reconnecting = true) else s
                }
                runSyncEngine() // flush what's queued
            }
        }

        shutterWatchJob?.cancel()
        // Blocking bulkTransfer I/O — must live on Dispatchers.IO (cancellation
        // is only observed at the controller's delay() points).
        shutterWatchJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                shutterWatchController.run(
                    catchUpAfter = catchUpAfter,
                    onHighWater = { handle -> tetherPrefs.edit().putLong(highWaterKey, handle).apply() },
                    onLog = { line -> appendWatchLog(line) },
                    onStarted = {
                        _shutterWatchState.value = ShutterWatchState.Watching(
                            captureCount = 0,
                            lastCaptureName = null,
                            recentLog = watchLogTail,
                        )
                    },
                    // Fires only after onStarted, so captureCount is never reset
                    // by a reconnect — the count is for the whole shoot.
                    onReconnecting = { reconnecting ->
                        _shutterWatchState.update { s ->
                            if (s is ShutterWatchState.Watching) s.copy(reconnecting = reconnecting) else s
                        }
                    },
                    onCapture = { filename, writeTo ->
                        val ok = persistCapturedJpeg(filename, eventId, photographerId, writeTo)
                        if (ok) {
                            _shutterWatchState.update { s ->
                                if (s is ShutterWatchState.Watching) {
                                    s.copy(captureCount = s.captureCount + 1, lastCaptureName = filename)
                                } else s
                            }
                            runSyncEngine() // per-frame kick; unique KEEP makes this free
                        }
                        ok
                    },
                )
            } finally {
                // Runs on cancellation too — everything here is non-suspending.
                // runSyncEngine() must be called directly: a withContext(Main)
                // would throw inside a cancelled coroutine.
                // Any orderly end (Stop, logout, the controller giving up) clears
                // the resume flag; only a dead process leaves it behind.
                tetherPrefs.edit().remove(WATCH_ACTIVE_EVENT_KEY).apply()
                shutterWatchDetachJob?.cancel()
                shutterWatchDetachJob = null
                _shutterWatchState.update { s ->
                    when (s) {
                        // stopShutterWatch/detach set Idle/Error BEFORE cancelling,
                        // so reaching here in Starting/Watching means the controller
                        // ended on its own — surface that instead of a silent reset.
                        is ShutterWatchState.Starting -> ShutterWatchState.Error(
                            "The camera isn't answering over USB. Switch the camera off and on " +
                                "(a replug alone doesn't clear this), wait a few seconds, then try again.",
                            watchLogTail,
                        )
                        is ShutterWatchState.Watching -> ShutterWatchState.Error(
                            "Auto-upload stopped unexpectedly — check the cable and start again. Pulled photos keep uploading.",
                            watchLogTail,
                        )
                        else -> s
                    }
                }
                runSyncEngine() // end-of-session flush
            }
        }
    }

    fun stopShutterWatch() {
        // Idle FIRST: the job's finally reads this state after cancellation is
        // delivered, so ordering it before cancel() is what makes a manual stop
        // land on Idle instead of the finally's "stopped unexpectedly" Error.
        _shutterWatchState.value = ShutterWatchState.Idle
        // Observed at the controller's next delay(); an in-flight bulkTransfer
        // finishes or times out (≤5 s) first.
        shutterWatchJob?.cancel()
        shutterWatchJob = null
        shutterWatchDetachJob?.cancel()
        shutterWatchDetachJob = null
        runSyncEngine() // flush anything persisted but not yet kicked
    }

    /**
     * Fold a controller log line into the rolling tail (and live state), and
     * mirror it to logcat — the R6 verification protocol reads the full stream
     * via `adb logcat -s QP/TETHER` while the screen shows only the tail.
     */
    private fun appendWatchLog(line: String) {
        if (BuildConfig.DEBUG) android.util.Log.i("QP/TETHER", line)
        watchLogTail = (watchLogTail + line).takeLast(WATCH_LOG_LINES)
        _shutterWatchState.update { s ->
            if (s is ShutterWatchState.Watching) s.copy(recentLog = watchLogTail) else s
        }
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

    private val _settingsActionState = MutableStateFlow<String?>(null)
    val settingsActionState: StateFlow<String?> = _settingsActionState

    private val _isSavingSettings = MutableStateFlow(false)
    val isSavingSettings: StateFlow<Boolean> = _isSavingSettings

    fun clearSettingsActionState() {
        _settingsActionState.value = null
    }

    private suspend fun readSettingsAsset(uri: String): ByteArray? =
        withContext(Dispatchers.IO) {
            getApplication<Application>().contentResolver.openInputStream(Uri.parse(uri))
                ?.use { it.readAtMost(MAX_UPLOAD_BYTES + 1) }
        }

    // Returns the validated bytes, or null after setting the error state.
    private suspend fun readValidatedAsset(uri: String, label: String): ByteArray? {
        val bytes = readSettingsAsset(uri)
        if (bytes == null || bytes.isEmpty() || bytes.size > MAX_UPLOAD_BYTES) {
            _settingsActionState.value = "Error: $label must be a readable image up to 8 MB."
            return null
        }
        return bytes
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
        avatarUri: String?,
        coverUri: String?,
        watermarkUri: String?,
        // Wire value from ALLOWED_BRAND_COLORS (none/fresh/amber/indigo/rose/
        // ink). Null = leave whatever is on the server untouched.
        brandColor: String? = null,
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
                // 0. Read + validate every picked image BEFORE the first PATCH
                // — an unreadable or oversized pick must not abort a
                // half-applied save.
                val avatarBytes = avatarUri?.let { readValidatedAsset(it, "Avatar") ?: return@launch }
                val coverBytes = coverUri?.let { readValidatedAsset(it, "Cover") ?: return@launch }
                val watermarkBytes = watermarkUri?.let { readValidatedAsset(it, "Watermark") ?: return@launch }

                // 1. Update Brand (Name & Bio). brandColor passes through the
                // hydrated value — sending a literal "none" here silently reset
                // a brand colour the photographer had picked on the website on
                // every mobile save. (A mobile picker is separate work; until
                // then mobile must at least not destroy the setting.)
                val brandResponse = RetrofitClient.apiService.updateBrand(
                    "Bearer $token",
                    com.quickpitik.mobile.data.remote.BrandPatchRequest(
                        brandName = brandName,
                        brandColor = brandColor ?: brandSettings.value?.brandColor ?: "none",
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

                // 2. Update GCash Payout — checked like every other step in
                // this chain. Ignoring the response let a rejected payout
                // account (bad number format, duplicate) report overall
                // success while the account was never saved.
                if (gcashName.isNotBlank() && gcashNumber.isNotBlank()) {
                    val payoutResponse = RetrofitClient.apiService.createPayoutAccount(
                        "Bearer $token",
                        com.quickpitik.mobile.data.remote.CreatePayoutRequest(
                            method = "gcash",
                            accountNumber = gcashNumber,
                            accountName = gcashName
                        )
                    )
                    if (!payoutResponse.success) {
                        _settingsActionState.value =
                            "Error: " + (payoutResponse.error ?: "Failed to save payout account.")
                        _isSavingSettings.value = false
                        return@launch
                    }
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
            } catch (e: retrofit2.HttpException) {
                // The backend's envelope is {success, errors:[{code,message}]} —
                // the previous flat {"error"/"message"} parse never matched it,
                // so HANDLE_TAKEN / RESERVED_HANDLE / INVALID_REGION /
                // VALIDATION_ERROR all collapsed to a hardcoded 409 guess.
                val err = RetrofitClient.parseHttpError(e)
                _settingsActionState.value = "Error: " + (err?.message
                    ?: if (e.code() == 409) "That handle is already taken by another user."
                    else (e.localizedMessage ?: "Connection error."))
            } catch (e: Exception) {
                _settingsActionState.value = "Error: " + RetrofitClient.parseError(e)
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

    // ── Socials immediate-CRUD ────────────────────────────────────────────
    fun addSocialAccount(platform: String, url: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _settingsActionState.value = "Saving social..."
            try {
                val response = RetrofitClient.apiService.createSocial(
                    "Bearer $token",
                    com.quickpitik.mobile.data.remote.CreateSocialRequest(platform, url),
                )
                if (response.success) {
                    _settingsActionState.value = "Success: Social added."
                    fetchSettings()
                } else {
                    _settingsActionState.value = "Error: " + (response.error ?: "Failed to add social.")
                }
            } catch (e: Exception) {
                _settingsActionState.value = "Error: " + (e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun updateSocialAccount(id: String, url: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _settingsActionState.value = "Updating social..."
            try {
                val response = RetrofitClient.apiService.patchSocial(
                    "Bearer $token",
                    id,
                    com.quickpitik.mobile.data.remote.PatchSocialRequest(url),
                )
                if (response.success) {
                    _settingsActionState.value = "Success: Social updated."
                    fetchSettings()
                } else {
                    _settingsActionState.value = "Error: " + (response.error ?: "Failed to update social.")
                }
            } catch (e: Exception) {
                _settingsActionState.value = "Error: " + (e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun deleteSocialAccount(id: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _settingsActionState.value = "Removing social..."
            try {
                val response = RetrofitClient.apiService.deleteSocial("Bearer $token", id)
                if (response.success) {
                    _settingsActionState.value = "Success: Social removed."
                    fetchSettings()
                } else {
                    _settingsActionState.value = "Error: " + (response.error ?: "Failed to remove social.")
                }
            } catch (e: Exception) {
                _settingsActionState.value = "Error: " + (e.localizedMessage ?: "Connection error.")
            }
        }
    }

    // ── Payouts immediate-CRUD ────────────────────────────────────────────
    fun addPayoutAccount(
        method: String,
        accountName: String,
        accountNumber: String,
        qrUri: String? = null,
    ) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _settingsActionState.value = "Saving payout..."
            try {
                val qrBytes = qrUri?.let { readValidatedAsset(it, "QR") ?: return@launch }
                val createResponse = RetrofitClient.apiService.createPayoutAccount(
                    "Bearer $token",
                    com.quickpitik.mobile.data.remote.CreatePayoutRequest(method, accountNumber, accountName),
                )
                if (!createResponse.success || createResponse.data == null) {
                    _settingsActionState.value = "Error: " + (createResponse.error ?: "Failed to add payout.")
                    return@launch
                }
                if (qrBytes != null) {
                    val newId = createResponse.data.id
                    if (!uploadPayoutQr(token, newId, qrBytes)) {
                        // The account row DID land — refresh so it shows and a
                        // retry doesn't create a duplicate.
                        _settingsActionState.value = "Error: Payout was added, but its QR could not be uploaded."
                        fetchSettings()
                        return@launch
                    }
                }
                _settingsActionState.value = "Success: Payout added."
                fetchSettings()
            } catch (e: Exception) {
                _settingsActionState.value = "Error: " + (e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun updatePayoutAccount(
        id: String,
        accountName: String,
        accountNumber: String,
        qrUri: String? = null,
    ) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _settingsActionState.value = "Updating payout..."
            try {
                val qrBytes = qrUri?.let { readValidatedAsset(it, "QR") ?: return@launch }
                val response = RetrofitClient.apiService.patchPayout(
                    "Bearer $token",
                    id,
                    com.quickpitik.mobile.data.remote.PatchPayoutRequest(accountNumber, accountName),
                )
                if (response.success) {
                    if (qrBytes != null && !uploadPayoutQr(token, id, qrBytes)) {
                        // The patch DID land — refresh so the list reflects it.
                        _settingsActionState.value = "Error: Payout was updated, but its QR could not be uploaded."
                        fetchSettings()
                        return@launch
                    }
                    _settingsActionState.value = "Success: Payout updated."
                    fetchSettings()
                } else {
                    _settingsActionState.value = "Error: " + (response.error ?: "Failed to update payout.")
                }
            } catch (e: Exception) {
                _settingsActionState.value = "Error: " + (e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun deletePayoutAccount(id: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _settingsActionState.value = "Removing payout..."
            try {
                val response = RetrofitClient.apiService.deletePayout("Bearer $token", id)
                if (response.success) {
                    _settingsActionState.value = "Success: Payout removed."
                    fetchSettings()
                } else {
                    _settingsActionState.value = "Error: " + (response.error ?: "Failed to remove payout.")
                }
            } catch (e: Exception) {
                _settingsActionState.value = "Error: " + (e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun setPrimaryPayoutAccount(id: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _settingsActionState.value = "Updating primary…"
            try {
                val response = RetrofitClient.apiService.setPrimaryPayout("Bearer $token", id)
                if (response.success && response.data != null) {
                    _payoutAccounts.value = response.data
                    _settingsActionState.value = "Success: Primary updated."
                } else {
                    _settingsActionState.value = "Error: " + (response.error ?: "Failed to set primary.")
                }
            } catch (e: Exception) {
                _settingsActionState.value = "Error: " + (e.localizedMessage ?: "Connection error.")
            }
        }
    }

    private suspend fun uploadPayoutQr(token: String, id: String, bytes: ByteArray): Boolean {
        val request = bytes.toRequestBody("image/png".toMediaTypeOrNull(), 0, bytes.size)
        val part = MultipartBody.Part.createFormData("file", "qr.png", request)
        // A thrown HTTP error (e.g. 429 — the QR route shares the backend's
        // 20/min media-upload bucket) must land on the callers' "payout was
        // added, but…" branch and its fetchSettings(), not skip into the
        // generic catch. Both failure shapes → false; cancellation still
        // propagates.
        return try {
            RetrofitClient.apiService.uploadPayoutQr("Bearer $token", id, part).success
        } catch (e: CancellationException) {
            throw e
        } catch (_: Exception) {
            false
        }
    }

    private companion object {
        const val INBOX_CHANNEL = "/ws/me/photographer/notifications"
    }
}

