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
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.CameraRecoverySummary
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.StatusCount
import com.quickpitik.mobile.data.local.TetherEvents
import com.quickpitik.mobile.data.local.TetherStorageCleanup
import com.quickpitik.mobile.data.local.TetherWatchPrefs
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.data.local.UploadSpool
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.data.usb.CameraConnectionState
import com.quickpitik.mobile.data.usb.CameraIngestCoordinator
import com.quickpitik.mobile.data.usb.ShutterWatchState
import com.quickpitik.mobile.data.usb.ptp.CardPhoto
import com.quickpitik.mobile.data.usb.ptp.UsbCardBrowseController
import com.quickpitik.mobile.data.usb.ptp.UsbCardImportController
import com.quickpitik.mobile.ui.common.canUploadToEvent
import com.quickpitik.mobile.worker.PhotoUploadScheduler
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale


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

/**
 * The tethered camera and everything downstream of the shutter: which event is
 * being shot, the USB connection, card browse and import, the live shutter
 * watch, the upload queue's counters, and the phone-storage cleanup.
 *
 * Split out of [PhotographerDashboardViewModel] (plan 2 phase G), on the same
 * "studio" nav graph entry - so a live tether session is torn down by exactly
 * what tore it down before: logout, or switching to the runner view, popping
 * the graph.
 *
 * [knownEvents] is the dashboard's event list, and the resume path is the only
 * thing that reads it: after process death the saved snapshot is normally
 * enough, but when it is missing or names a different event (a build from
 * before the snapshot existed, or a crash between the two preference writes)
 * the event is looked up in the list instead. A defaulted parameter rather
 * than a reach into the other ViewModel, so a test hands it a flow directly.
 */
class TetherViewModel(
    application: Application,
    private val knownEvents: StateFlow<EventsState> = MutableStateFlow(EventsState.Loading),
) : AndroidViewModel(application) {
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
        observeQueue()
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
            val ready = combine(knownEvents, cameraManager.state) { events, cam ->
                (saved ?: (events as? EventsState.Success)?.events?.firstOrNull { it.id == eventId })
                    ?.takeIf { cam is CameraConnectionState.Connected }
            }.first { it != null } ?: return@launch
            if (tetherPrefs.activeEventId != eventId ||
                tetherPrefs.owner != sessionManager.getUserEmail()) return@launch
            _activeEvent.value = ready
            if (_shutterWatchState.value is ShutterWatchState.Idle) startShutterWatch()
        }
    }

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

    fun refreshCameraConnection() {
        cameraManager.refresh()
    }

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
