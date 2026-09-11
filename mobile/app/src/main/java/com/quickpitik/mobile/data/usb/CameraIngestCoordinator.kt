package com.quickpitik.mobile.data.usb

import android.content.Context
import android.content.Intent
import androidx.core.content.ContextCompat
import com.quickpitik.mobile.BuildConfig
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.CameraTransferStore
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.TetherWatchPrefs
import com.quickpitik.mobile.data.usb.ptp.UsbEventCaptureController
import com.quickpitik.mobile.service.TetherIngestService
import com.quickpitik.mobile.worker.PhotoUploadScheduler
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

sealed class ShutterWatchState {
    object Idle : ShutterWatchState()
    object Starting : ShutterWatchState()
    object Stopping : ShutterWatchState()
    data class Watching(
        val captureCount: Int,
        val lastCaptureName: String?,
        val recentLog: List<String>,
        val reconnecting: Boolean = false,
        val pauseReason: String? = null,
    ) : ShutterWatchState()
    data class Error(val message: String, val recentLog: List<String> = emptyList()) : ShutterWatchState()
}

/** App-visible state; all USB work is a child of the foreground service's scope. */
class CameraIngestCoordinator private constructor(private val context: Context) {
    val cameraManager = CameraConnectionManager(context)
    val store = CameraTransferStore(context, AppDatabase.getDatabase(context))
    private val session = SessionManager.getInstance(context)
    private val prefs = TetherWatchPrefs(context)
    val watch = MutableStateFlow<ShutterWatchState>(ShutterWatchState.Idle)
    val cardBusy = MutableStateFlow(false)
    private var cardStart: CompletableDeferred<CoroutineScope>? = null
    private var cardGeneration = 0L
    @Volatile private var watchJob: Job? = null
    @Volatile private var generation = 0L
    private var runningGeneration = -1L
    private var logs = emptyList<String>()
    val isWatching get() = watch.value is ShutterWatchState.Starting || watch.value is ShutterWatchState.Watching || watch.value is ShutterWatchState.Stopping

    fun attach() { cameraManager.start() }
    fun detach() {
        if (isWatching && runningGeneration == generation && watch.value !is ShutterWatchState.Stopping) {
            generation++
            watch.value = ShutterWatchState.Error("Camera service stopped. Reopen QuickPitik to resume pending photos.", logs)
        }
    }

    fun requestWatch(owner: String, event: String) {
        if (isWatching || cardBusy.value || watchJob?.isCompleted == false) return
        generation++
        logs = emptyList()
        watch.value = ShutterWatchState.Starting
        prefs.save(owner = owner, eventId = event)
        val intent = Intent(context, TetherIngestService::class.java).setAction(TetherIngestService.ACTION_WATCH)
            .putExtra("owner", owner).putExtra("event", event).putExtra("generation", generation)
        try { ContextCompat.startForegroundService(context, intent) }
        catch (e: RuntimeException) { startFailed() }
    }

    fun startFailed() {
        generation++
        watch.value = ShutterWatchState.Error("Allow camera USB access and keep QuickPitik open, then start again.", logs)
    }

    fun beginWatch(scope: CoroutineScope, owner: String, event: String, request: Long) {
        if (request != generation || watch.value !is ShutterWatchState.Starting || watchJob?.isActive == true) return
        runningGeneration = request
        watchJob = scope.launch(Dispatchers.IO) {
            try {
                UsbEventCaptureController(cameraManager.manager, cameraManager::connectedCameraDevice, store,
                    awaitDevice = { timeout -> withTimeoutOrNull(timeout) {
                        cameraManager.state.first { it is CameraConnectionState.Connected && cameraManager.connectedCameraDevice() != null }
                    }; Unit },
                ).run(
                    owner, event, isOwnerActive = { session.getUserEmail() == owner && session.getAccessToken() != null },
                    onLog = { if (generation == request) log(it) },
                    onStarted = { if (generation == request) watch.value = ShutterWatchState.Watching(0, null, logs) },
                    onReconnecting = { reconnecting -> if (generation == request) watch.update {
                        when (it) {
                            is ShutterWatchState.Watching -> it.copy(reconnecting = reconnecting)
                            ShutterWatchState.Starting -> if (reconnecting) ShutterWatchState.Watching(0, null, logs, true) else it
                            else -> it
                        }
                    } },
                    onPaused = { reason -> if (generation == request) watch.update { if (it is ShutterWatchState.Watching) it.copy(pauseReason = reason) else it } },
                    onCapture = { name ->
                        if (generation == request) watch.update { if (it is ShutterWatchState.Watching) it.copy(captureCount = it.captureCount + 1, lastCaptureName = name) else it }
                        PhotoUploadScheduler.enqueue(context)
                    },
                )
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                if (generation == request) watch.value = ShutterWatchState.Error(e.message ?: "Camera transfer interrupted.", logs)
            } finally { PhotoUploadScheduler.enqueue(context) }
        }
    }

    fun stopWatch() {
        generation++
        prefs.clear()
        val previous = watchJob
        if (previous == null || previous.isCompleted) { watch.value = ShutterWatchState.Idle; return }
        watch.value = ShutterWatchState.Stopping
        previous.cancel()
        // Completion runs only after the camera driver's finally has released the USB interface.
        previous.invokeOnCompletion {
            if (watchJob === previous) { watchJob = null; watch.value = ShutterWatchState.Idle }
        }
    }

    fun beginCard(scope: CoroutineScope, request: Long) {
        if (request == cardGeneration) cardStart?.complete(scope)
    }

    suspend fun runCard(block: suspend () -> Unit) = withContext(Dispatchers.Main.immediate) {
        check(!isWatching) { "Stop auto-upload before opening the card." }
        check(cardBusy.compareAndSet(false, true)) { "The camera is finishing another operation. Try again shortly." }
        val ready = CompletableDeferred<CoroutineScope>()
        cardStart = ready
        cardGeneration++
        try {
            ContextCompat.startForegroundService(context,
                Intent(context, TetherIngestService::class.java).setAction(TetherIngestService.ACTION_START)
                    .putExtra("card_generation", cardGeneration)
                    .putExtra(TetherIngestService.EXTRA_NOTICE, "Reading camera card..."))
            // Wait for this start command, never borrow a service scope already being stopped.
            val scope = withTimeout(10_000) { ready.await() }
            val task = scope.async(Dispatchers.IO) { block() }
            try { task.await() } finally { withContext(NonCancellable) { task.cancelAndJoin() } }
        } finally { cardStart = null; cardBusy.value = false }
    }

    private fun log(line: String) {
        if (BuildConfig.DEBUG) android.util.Log.i("QP/TETHER", line)
        logs = (logs + line).takeLast(12)
        watch.update { if (it is ShutterWatchState.Watching) it.copy(recentLog = logs) else it }
    }

    companion object {
        @Volatile private var instance: CameraIngestCoordinator? = null
        fun get(context: Context): CameraIngestCoordinator = instance ?: synchronized(this) {
            instance ?: CameraIngestCoordinator(context.applicationContext).also { instance = it }
        }
    }
}
