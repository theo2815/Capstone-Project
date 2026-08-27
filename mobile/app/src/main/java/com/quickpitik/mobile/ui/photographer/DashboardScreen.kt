package com.quickpitik.mobile.ui.photographer

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*

import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.data.remote.PhotographerMessageDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.usb.CameraConnectionState
import com.quickpitik.mobile.ui.runner.EventState
import com.quickpitik.mobile.ui.runner.canUploadToEvent
import com.quickpitik.mobile.ui.runner.deriveEventState
import com.quickpitik.mobile.ui.runner.eventDateLabel
import com.quickpitik.mobile.ui.runner.extractCity
import com.quickpitik.mobile.ui.theme.*
import java.io.File

/**
 * The Capture tab's content: event picker until an event is selected, then the
 * tether console. Extracted from the old dashboard's `when(currentTab)` branch
 * when the tabs became NavHost routes — the shell chrome now lives in
 * StudioShell.kt; this file keeps the tether/capture surfaces it always owned.
 *
 * PublicEventPickerList's own LaunchedEffect refires on re-entry (identical to
 * the old tab behavior), so `studio/capture` is deliberately NOT in the shared
 * per-tab refetch — adding it would double the fetch.
 */
@Composable
fun PhotographerCaptureScreen(viewModel: PhotographerDashboardViewModel) {
    val activeEvent by viewModel.activeEvent.collectAsState()
    if (activeEvent == null) {
        PublicEventPickerList(
            viewModel = viewModel,
            onSelectEvent = { event -> viewModel.selectEvent(event) }
        )
    } else {
        TetherConsoleView(viewModel = viewModel)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TetherConsoleView(
    viewModel: PhotographerDashboardViewModel,
    modifier: Modifier = Modifier
) {
    val activeEvent by viewModel.activeEvent.collectAsState()
    val queueStats by viewModel.queueStats.collectAsState()
    val cameraState by viewModel.cameraConnectionState.collectAsState()
    val watchState by viewModel.shutterWatchState.collectAsState()
    val scrollState = rememberScrollState()

    // Per-photo strip data: this event's rows only, newest first, capped so a
    // 500-shot ledger doesn't become 500 tiles. Everything pending anywhere
    // (any event) still keeps the sync card visible below.
    val recentForEvent = remember(queueStats.recentRecords, activeEvent?.id) {
        queueStats.recentRecords.asSequence()
            .filter { it.eventId == activeEvent?.id }
            .take(RECENT_STRIP_MAX_TILES)
            .toList()
    }
    val pendingWork = queueStats.queuedCount + queueStats.uploadingCount + queueStats.failedCount

    // Android 13+ suppresses TetherIngestService's notification unless
    // POST_NOTIFICATIONS is granted. The ingest itself runs either way — what's
    // lost is the shade's live status and Stop button — so the prompt rides
    // alongside the first ingest instead of gating it. Asked once per mount: a
    // photographer who declined shouldn't be re-prompted on every start
    // mid-race.
    val context = LocalContext.current
    var notificationsDenied by remember { mutableStateOf(false) }
    var notificationAsked by remember { mutableStateOf(false) }
    val notificationPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> notificationsDenied = !granted }
    val ensureNotificationPermission = {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && !notificationAsked) {
            notificationAsked = true
            val granted = ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.POST_NOTIFICATIONS,
            ) == PackageManager.PERMISSION_GRANTED
            if (!granted) notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Bone)
            .verticalScroll(scrollState)
            .padding(horizontal = 20.dp, vertical = 16.dp)
            .navigationBarsPadding(),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        // 1. Back affordance + active-event header.
        Column(modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier
                    .clickable { viewModel.selectEvent(null) }
                    .heightIn(min = 48.dp)
                    .padding(vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.ArrowBack,
                    contentDescription = null,
                    tint = Slate,
                    modifier = Modifier.size(14.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Kicker(text = "Pick another event", color = Slate)
            }
            Spacer(modifier = Modifier.height(12.dp))

            val stateLabel = activeEvent?.state?.uppercase() ?: "EVENT"
            val tone = when (stateLabel) {
                "LIVE", "OPEN" -> StatusTone.Approved
                // Neutral — upcoming isn't a caution state (web parity).
                else -> StatusTone.Neutral
            }
            StatusChip(text = "Event · $stateLabel", tone = tone)
            Spacer(modifier = Modifier.height(10.dp))
            Text(
                text = activeEvent?.name ?: "No active event",
                color = Ink,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                lineHeight = 30.sp
            )
            activeEvent?.let { ev ->
                Spacer(modifier = Modifier.height(4.dp))
                val cityLabel = extractCity(ev.location)
                Text(
                    text = "${ev.date} · ${if (cityLabel.isNotBlank()) cityLabel else ev.location}",
                    color = Slate,
                    style = Typography.bodyMedium
                )
            }
        }

        // 2. Camera connection. Branches on the real USB-host state from
        // CameraConnectionManager. Mobile is tether-only — gallery upload lives
        // on the website (`/dashboard/upload`), not here.
        when (val cam = cameraState) {
            // Detach edge (accepted): unplugging mid-watch swaps this branch to
            // CameraConnectPrompt, hiding the VM's shutter-watch Error until the
            // camera is re-plugged. The detach watcher still stops the session
            // and flushes the queue.
            is CameraConnectionState.Connected -> CameraConnectedCard(
                deviceName = cam.deviceName,
                vendorId = cam.vendorId,
                productId = cam.productId,
                watchState = watchState,
                latestCaptureFile = recentForEvent.firstOrNull()?.let { File(it.filePath) },
                canStartWatch = activeEvent?.let { canUploadToEvent(it.date) } == true,
                onStartWatch = {
                    ensureNotificationPermission()
                    viewModel.startShutterWatch()
                },
                onStopWatch = { viewModel.stopShutterWatch() },
                onSimulate = { viewModel.simulatePhotoCapture() },
                onBrowseCard = {
                    ensureNotificationPermission()
                    viewModel.browseCameraCard()
                }
            )
            CameraConnectionState.Searching,
            CameraConnectionState.Disconnected -> CameraConnectPrompt(
                isSearching = cam is CameraConnectionState.Searching,
                onRescan = { viewModel.refreshCameraConnection() }
            )
        }

        if (notificationsDenied) NotificationsOffNote()

        // 2b. Manual card-import sheet — mounted whenever a browse is live.
        // Increment 1 reads; Increment 2 wires selection; Increment 3 pulls
        // bytes off the card; Increment 4 adds dedupe + retry-failed + cancel +
        // live camera-disconnect detection. Existing PhotoUploadWorker handles
        // the S3 leg unchanged.
        val cardBrowse by viewModel.cardBrowseState.collectAsState()
        if (cardBrowse !is CardBrowseState.Idle) {
            CameraCardImportSheet(
                state = cardBrowse,
                onDismiss = { viewModel.closeCardImport() },
                onRetry = { viewModel.browseCameraCard() },
                onToggleSelect = { handle -> viewModel.toggleCardPhotoSelection(handle) },
                onSelectAll = { viewModel.selectAllCardPhotos() },
                onClearSelection = { viewModel.clearCardPhotoSelection() },
                onImport = { viewModel.importSelectedCardPhotos() },
                onCancelImport = { viewModel.closeCardImport() },
                onRetryFailed = { viewModel.retryFailedCardImports() },
            )
        }

        // 3. Sync queue — visible while a camera is tethered, while anything is
        // still pending (so unplugging mid-drain doesn't hide the uploads), or
        // when this event has strip rows to show. Deliberately NOT
        // totalCount > 0: COMPLETED rows are a permanent dedupe ledger, which
        // would pin the card open on every event forever.
        if (
            cameraState is CameraConnectionState.Connected ||
            pendingWork > 0 ||
            recentForEvent.isNotEmpty()
        ) {
        QpCard(modifier = Modifier.fillMaxWidth()) {
            Kicker(text = "Sync queue", color = SlateSoft)
            Spacer(modifier = Modifier.height(14.dp))
            LinearProgressIndicator(
                progress = queueStats.progress,
                color = Slate,
                trackColor = Line,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(PillShape)
            )
            Spacer(modifier = Modifier.height(14.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = queueStats.syncedCount.toString(),
                        style = NumeralStyle.copy(fontSize = 22.sp),
                        color = Ink
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Kicker(text = "Synced", color = SlateSoft)
                }
                val (statusText, statusTone) = when {
                    queueStats.uploadingCount > 0 ->
                        "Uploading · ${queueStats.uploadingCount}" to StatusTone.Approved
                    queueStats.queuedCount > 0 ->
                        "Queued · ${queueStats.queuedCount}" to StatusTone.Warning
                    queueStats.failedCount > 0 ->
                        "Failed · ${queueStats.failedCount}" to StatusTone.Danger
                    else -> "Idle" to StatusTone.Neutral
                }
                StatusChip(text = statusText, tone = statusTone)
            }
            // Per-photo strip: the frames themselves, newest first, each with
            // its own status badge — the "did the right photo make it?" answer
            // the counts alone never gave.
            AnimatedVisibility(
                visible = recentForEvent.isNotEmpty(),
                enter = fadeIn(tween(200)),
                exit = fadeOut(tween(200)),
            ) {
                LazyRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 14.dp),
                ) {
                    items(recentForEvent, key = { it.id }) { record ->
                        SyncQueueTile(record)
                    }
                }
            }
            // Surface the most recent failure string so a "Failed · N" chip
            // isn't a dead end. PhotoUploadWorker already stores the per-row
            // error (auth, timeout, server reject) — we just hadn't exposed
            // it. Small subdued line, capped to 2 lines, no banner — keeps
            // the Quiet Studio quietness of the card.
            queueStats.lastError?.let { errorMessage ->
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = "Last error: $errorMessage",
                    style = MaterialTheme.typography.bodySmall,
                    color = SlateSoft,
                    maxLines = 2,
                    overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                )
            }
            Spacer(modifier = Modifier.height(16.dp))
            GhostCta(
                text = if (queueStats.uploadingCount > 0) "Syncing…" else "Run sync engine",
                onClick = { viewModel.runSyncEngine() },
                modifier = Modifier.fillMaxWidth(),
                enabled = queueStats.uploadingCount == 0
            )
            // Retry + Clear only appear when there's actually something failed
            // — keeps the card quiet on the happy path. Both disabled mid-
            // upload so the actions can't race with a running worker.
            if (queueStats.failedCount > 0) {
                Spacer(modifier = Modifier.height(10.dp))
                GhostCta(
                    text = "Retry ${queueStats.failedCount} failed",
                    onClick = { viewModel.retryFailedUploads() },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = queueStats.uploadingCount == 0
                )
                Spacer(modifier = Modifier.height(10.dp))
                GhostCta(
                    text = "Clear ${queueStats.failedCount} failed",
                    onClick = { viewModel.clearFailedUploads() },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = queueStats.uploadingCount == 0
                )
            }
        }
        }
    }
}

@Composable
private fun CameraConnectPrompt(
    isSearching: Boolean,
    onRescan: () -> Unit,
) {
    QpCard(modifier = Modifier.fillMaxWidth()) {
        StatusChip(
            text = if (isSearching) "Camera · searching" else "Camera · waiting",
            tone = if (isSearching) StatusTone.Warning else StatusTone.Neutral
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "Connect your camera",
            color = Ink,
            fontSize = 18.sp,
            fontWeight = FontWeight.SemiBold,
            lineHeight = 24.sp
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = "Mobile uploads run through a tethered camera. Phone-gallery uploads live on the website.",
            color = Slate,
            style = Typography.bodyMedium,
            lineHeight = 20.sp
        )
        Spacer(modifier = Modifier.height(20.dp))

        // USB-C row — actionable path.
        ConnectMethodRow(
            method = "USB-C",
            body = "Plug your camera into your phone with a USB-C cable. We'll detect it automatically and ask for permission.",
            trailing = {
                StatusChip(
                    text = if (isSearching) "Searching" else "Ready",
                    tone = if (isSearching) StatusTone.Warning else StatusTone.Approved
                )
            }
        )
        Spacer(modifier = Modifier.height(12.dp))
        Divider(color = Line)
        Spacer(modifier = Modifier.height(12.dp))

        // Wi-Fi row — coming soon.
        ConnectMethodRow(
            method = "Wi-Fi",
            body = "Wireless FTP tethering — point your camera's built-in FTP at the phone. In development.",
            trailing = {
                StatusChip(text = "Coming soon", tone = StatusTone.Neutral)
            }
        )

        Spacer(modifier = Modifier.height(20.dp))
        GhostCta(
            text = "Re-scan USB",
            onClick = onRescan,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

@Composable
private fun ConnectMethodRow(
    method: String,
    body: String,
    trailing: @Composable () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.Top,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f).padding(end = 12.dp)) {
            Kicker(text = method, color = Ink)
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = body,
                color = Slate,
                style = Typography.bodyMedium,
                lineHeight = 20.sp
            )
        }
        trailing()
    }
}

@Composable
private fun CameraConnectedCard(
    deviceName: String,
    vendorId: Int,
    productId: Int,
    watchState: ShutterWatchState,
    canStartWatch: Boolean,
    onStartWatch: () -> Unit,
    onStopWatch: () -> Unit,
    onSimulate: () -> Unit,
    onBrowseCard: () -> Unit,
    latestCaptureFile: File? = null,
) {
    val watchBusy = watchState is ShutterWatchState.Starting ||
        watchState is ShutterWatchState.Watching
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        QpCard(modifier = Modifier.fillMaxWidth()) {
            StatusChip(text = "Camera · ready", tone = StatusTone.Approved)
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = deviceName,
                color = Ink,
                fontSize = 20.sp,
                fontWeight = FontWeight.SemiBold,
                lineHeight = 26.sp
            )
            Spacer(modifier = Modifier.height(4.dp))
            val vidHex = vendorId.toString(16).uppercase().padStart(4, '0')
            val pidHex = productId.toString(16).uppercase().padStart(4, '0')
            Text(
                text = "USB · $vidHex:$pidHex",
                color = Slate,
                style = Typography.bodyMedium
            )
        }

        // Live auto-upload slot: Idle/Error offer the start CTA (the single
        // Fresh in this viewport); Watching swaps to the live-session card.
        AnimatedContent(
            targetState = watchState,
            contentKey = { it::class },
            transitionSpec = { fadeIn() togetherWith fadeOut() },
            label = "shutterWatch",
        ) { state ->
            when (state) {
                is ShutterWatchState.Watching -> ShutterWatchLiveCard(
                    state = state,
                    onStop = onStopWatch,
                    latestCaptureFile = latestCaptureFile,
                )
                else -> Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    if (state is ShutterWatchState.Error) {
                        QpCard(modifier = Modifier.fillMaxWidth()) {
                            Kicker(text = "Auto-upload", color = ErrorRed)
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = state.message,
                                color = Slate,
                                style = Typography.bodyMedium,
                                lineHeight = 20.sp
                            )
                        }
                    }
                    PrimaryCta(
                        text = "Start auto-upload",
                        onClick = onStartWatch,
                        modifier = Modifier.fillMaxWidth(),
                        enabled = canStartWatch && state !is ShutterWatchState.Starting,
                        loading = state is ShutterWatchState.Starting,
                    )
                    if (!canStartWatch) {
                        Text(
                            text = "This event's upload window has closed.",
                            color = Slate,
                            style = Typography.bodyMedium
                        )
                    }
                }
            }
        }

        // Simulate stays as the wire-free pipeline exercise; it never touches
        // USB, so it's safe even while a live watch holds the session.
        GhostCta(
            text = "Simulate DSLR shoot",
            onClick = onSimulate,
            modifier = Modifier.fillMaxWidth()
        )
        // Manual card-import opens its OWN PTP session, which would force-claim
        // the interface out from under a live watch — disabled while watching.
        GhostCta(
            text = "Import from camera card",
            onClick = onBrowseCard,
            modifier = Modifier.fillMaxWidth(),
            enabled = !watchBusy
        )
    }
}

@Composable
private fun ShutterWatchLiveCard(
    state: ShutterWatchState.Watching,
    onStop: () -> Unit,
    latestCaptureFile: File? = null,
) {
    QpCard(modifier = Modifier.fillMaxWidth()) {
        // A dropped cable is a pause, not a failure — the count keeps standing
        // and the controller is already reopening the session, so this reads as
        // a state change on the same card rather than an error.
        if (state.reconnecting) {
            StatusChip(text = "Auto-upload · reconnecting", tone = StatusTone.Warning)
        } else {
            StatusChip(text = "Auto-upload · live", tone = StatusTone.Approved)
        }
        Spacer(modifier = Modifier.height(14.dp))
        Text(
            text = state.captureCount.toString(),
            style = NumeralStyle.copy(fontSize = 22.sp),
            color = Ink
        )
        Spacer(modifier = Modifier.height(2.dp))
        Kicker(text = "Captured this session", color = SlateSoft)
        if (state.reconnecting) {
            Spacer(modifier = Modifier.height(10.dp))
            Text(
                text = "Camera link dropped. Re-plug the USB cable — shots taken " +
                    "meanwhile upload once it reconnects.",
                color = Slate,
                style = Typography.bodyMedium,
            )
        }
        state.lastCaptureName?.let { name ->
            Spacer(modifier = Modifier.height(10.dp))
            // The "I just shot this and here it is" moment: the newest queue
            // row's spool file rendered beside the filename.
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (latestCaptureFile != null) {
                    Box(
                        modifier = Modifier
                            .size(48.dp)
                            .clip(TileShape)
                            .background(BoneDeep)
                    ) {
                        AsyncImage(
                            model = latestCaptureFile,
                            contentDescription = "Last capture",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize()
                        )
                    }
                    Spacer(modifier = Modifier.width(10.dp))
                }
                Text(
                    text = "Last capture: $name",
                    color = Slate,
                    style = Typography.bodyMedium,
                    maxLines = 1,
                    overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                )
            }
        }
        // Field-diagnostics tail — the controller's own log lines, so the first
        // on-device run reveals which detector fires without needing adb.
        if (state.recentLog.isNotEmpty()) {
            Spacer(modifier = Modifier.height(12.dp))
            Divider(color = Line)
            Spacer(modifier = Modifier.height(12.dp))
            state.recentLog.forEach { line ->
                Text(
                    text = line,
                    style = MaterialTheme.typography.bodySmall,
                    color = SlateSoft,
                    maxLines = 2,
                    overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                )
            }
        }
        Spacer(modifier = Modifier.height(16.dp))
        GhostCta(
            text = "Stop auto-upload",
            onClick = onStop,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

// Cap on the per-photo sync strip: newest N rows of the selected event. The
// COMPLETED ledger is unbounded; the strip is not.
private const val RECENT_STRIP_MAX_TILES = 30

/**
 * One frame in the sync strip. Pre-upload rows render the full spool JPEG
 * (Coil downsamples to tile size); COMPLETED rows render the worker's in-place
 * ~320px thumbnail. A missing/undecodable file (rows synced before the
 * shrink-on-success change) just leaves the BoneDeep tile. Per-tile
 * SuccessGreen/ErrorRed marks follow the card-import sheet's "Imported"
 * precedent — semantic state, not accent CTAs, so the one-Fresh rule holds.
 */
@Composable
private fun SyncQueueTile(record: UploadRecord) {
    val stateLabel = when (record.uploadStatus) {
        "UPLOADING" -> "uploading"
        "COMPLETED" -> "synced"
        "FAILED" -> "failed"
        else -> "queued"
    }
    Box(
        modifier = Modifier
            .size(64.dp)
            .clip(TileShape)
            .background(BoneDeep)
    ) {
        AsyncImage(
            model = File(record.filePath),
            contentDescription = "Photo $stateLabel",
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize()
        )
        // QUEUED gets no badge — the quiet default; it flips to a spinner
        // within seconds of the worker picking the row up.
        if (record.uploadStatus != "QUEUED") {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(3.dp)
                    .size(18.dp)
                    .clip(PillShape)
                    .background(Bone),
                contentAlignment = Alignment.Center,
            ) {
                when (record.uploadStatus) {
                    "UPLOADING" -> CircularProgressIndicator(
                        color = Slate,
                        strokeWidth = 2.dp,
                        modifier = Modifier.size(12.dp)
                    )
                    "COMPLETED" -> Icon(
                        Icons.Default.Check,
                        contentDescription = null,
                        tint = SuccessGreen,
                        modifier = Modifier.size(12.dp)
                    )
                    else -> Icon(
                        Icons.Default.Close,
                        contentDescription = null,
                        tint = ErrorRed,
                        modifier = Modifier.size(12.dp)
                    )
                }
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun SyncQueueTileStatesPreview() {
    // Nonexistent paths render the BoneDeep placeholder — the badge per state
    // is what this preview pins (queued / uploading / synced / failed).
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        listOf("QUEUED", "UPLOADING", "COMPLETED", "FAILED").forEach { status ->
            SyncQueueTile(
                UploadRecord(
                    id = status.hashCode().toLong(),
                    filePath = "/preview/no-such-file.jpg",
                    eventId = "preview-event",
                    photographerId = "preview@example.com",
                    captureTimestamp = 0L,
                    uploadStatus = status,
                )
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun CameraConnectedCardIdlePreview() {
    CameraConnectedCard(
        deviceName = "Canon Inc. Canon Digital Camera",
        vendorId = 0x04A9,
        productId = 0x32F5,
        watchState = ShutterWatchState.Idle,
        canStartWatch = true,
        onStartWatch = {},
        onStopWatch = {},
        onSimulate = {},
        onBrowseCard = {},
    )
}

@Preview(showBackground = true)
@Composable
private fun CameraConnectedCardErrorPreview() {
    CameraConnectedCard(
        deviceName = "Canon Inc. Canon Digital Camera",
        vendorId = 0x04A9,
        productId = 0x32F5,
        watchState = ShutterWatchState.Error(
            "Camera disconnected — auto-upload stopped. Photos already pulled keep uploading."
        ),
        canStartWatch = true,
        onStartWatch = {},
        onStopWatch = {},
        onSimulate = {},
        onBrowseCard = {},
    )
}

@Preview(showBackground = true)
@Composable
private fun CameraConnectedCardWatchingPreview() {
    CameraConnectedCard(
        deviceName = "Canon Inc. Canon Digital Camera",
        vendorId = 0x04A9,
        productId = 0x32F5,
        watchState = ShutterWatchState.Watching(
            captureCount = 12,
            lastCaptureName = "R6T_1083.JPG",
            recentLog = listOf(
                "event 0xC181 seen",
                "Capture via EVENT — R6T_1083.JPG (0x00000C4B)",
                "  R6T_1083.JPG 8412 KB → queued",
            ),
        ),
        canStartWatch = true,
        onStartWatch = {},
        onStopWatch = {},
        onSimulate = {},
        onBrowseCard = {},
    )
}

@Preview(showBackground = true)
@Composable
private fun CameraConnectedCardReconnectingPreview() {
    CameraConnectedCard(
        deviceName = "Canon Inc. Canon Digital Camera",
        vendorId = 0x04A9,
        productId = 0x32F5,
        watchState = ShutterWatchState.Watching(
            captureCount = 12,
            lastCaptureName = "R6T_1083.JPG",
            recentLog = listOf(
                "Camera link lost — retrying. Photos already pulled keep uploading.",
                "No camera / USB permission (attempt 1)…",
            ),
            reconnecting = true,
        ),
        canStartWatch = true,
        onStartWatch = {},
        onStopWatch = {},
        onSimulate = {},
        onBrowseCard = {},
    )
}

/**
 * Shown once the photographer declines POST_NOTIFICATIONS.
 *
 * Deliberately quiet — Slate body, no ErrorRed, no CTA. Nothing has failed:
 * the shoot runs exactly the same, they've just given up the shade's live
 * status and Stop button. Saying so beats leaving a missing notification
 * unexplained mid-race.
 */
@Composable
private fun NotificationsOffNote(modifier: Modifier = Modifier) {
    Text(
        text = "Notifications are off, so the shoot won't appear in your notification " +
            "shade. Photos still upload — start and stop from this screen.",
        color = Slate,
        style = Typography.bodyMedium,
        modifier = modifier.fillMaxWidth(),
    )
}

@Preview(showBackground = true)
@Composable
private fun NotificationsOffNotePreview() {
    NotificationsOffNote()
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PublicEventPickerList(
    viewModel: PhotographerDashboardViewModel,
    onSelectEvent: (PhotographerEventSummaryDto) -> Unit,
    modifier: Modifier = Modifier
) {
    val publicEventsState by viewModel.publicEventsState.collectAsState()
    val eventsState by viewModel.eventsState.collectAsState()
    val verificationState by viewModel.verificationState.collectAsState()
    val messages by viewModel.messages.collectAsState()

    var showPendingAlert by remember { mutableStateOf(false) }
    var showIncompleteAlert by remember { mutableStateOf(false) }

    val latestMessage = remember(messages) { messages.maxByOrNull { it.createdAt } }
    val currentStatus = (verificationState as? VerificationUiState.Success)?.verification?.status?.lowercase() ?: "incomplete"
    val isRejected = (currentStatus == "incomplete" || currentStatus == "rejected") && (latestMessage?.kind == "verification_rejected")

    val assignedEvents = when (val state = eventsState) {
        is EventsState.Success -> state.events
        else -> emptyList()
    }
    
    LaunchedEffect(Unit) {
        viewModel.fetchPublicEvents()
        viewModel.fetchEvents()
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Bone)
            .padding(horizontal = 20.dp, vertical = 16.dp)
            .navigationBarsPadding(),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Kicker(text = "Studio · capture", color = Slate)
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = "Pick an event",
                color = Ink,
                fontSize = 26.sp,
                fontWeight = FontWeight.Bold,
                lineHeight = 32.sp
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = "Choose a marathon below. Frames from your tethered camera sync to whichever event is active.",
                color = Slate,
                style = Typography.bodyMedium,
                lineHeight = 20.sp
            )
            Spacer(modifier = Modifier.height(10.dp))
            // Disclosure: explains why the photographer only sees a subset of
            // events here. Mirrors the backend's EVENT_NOT_UPLOADABLE window
            // (race day + 3 days). Past-window events still live in the
            // Events tab — they just can't accept new uploads.
            Text(
                text = "Only events inside the race-day + 3-day upload window appear here. Past that, the event closes for upload and moves to your Events tab.",
                color = SlateSoft,
                style = Typography.bodySmall,
                lineHeight = 16.sp
            )
        }

        when (val state = publicEventsState) {
            is EventsState.Loading -> {
                Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(color = Fresh)
                }
            }
            is EventsState.Error -> {
                Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                    ErrorView(
                        message = state.message,
                        title = "Couldn't load events",
                        onRetry = { viewModel.fetchPublicEvents() }
                    )
                }
            }
            is EventsState.Success -> {
                val allPickerEvents = (state.events + assignedEvents)
                    .distinctBy { it.id }
                    .sortedBy { it.date }

                if (allPickerEvents.isEmpty()) {
                    Box(
                        modifier = Modifier.fillMaxWidth().weight(1f).padding(24.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Kicker(text = "No events on the calendar", color = Slate)
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "Events created in the system will appear here for capture and upload.",
                                color = Slate,
                                style = Typography.bodyMedium,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                } else {
                    LazyColumn(
                        modifier = Modifier.fillMaxWidth().weight(1f),
                        verticalArrangement = Arrangement.spacedBy(14.dp),
                        contentPadding = PaddingValues(bottom = 24.dp)
                    ) {
                        items(allPickerEvents, key = { it.id }) { event ->
                            val canUpload = canUploadToEvent(event.date)
                            EventPickerCard(
                                event = event,
                                onClick = {
                                    if (canUpload) {
                                        when {
                                            currentStatus == "approved" -> onSelectEvent(event)
                                            isRejected -> showIncompleteAlert = true
                                            currentStatus == "pending" -> showPendingAlert = true
                                            else -> showIncompleteAlert = true
                                        }
                                    }
                                }
                            )
                        }
                    }
                }
            }
        }
    }

    if (showPendingAlert) {
        AlertDialog(
            onDismissRequest = { showPendingAlert = false },
            confirmButton = {
                Button(
                    onClick = { showPendingAlert = false },
                    colors = ButtonDefaults.buttonColors(containerColor = Fresh)
                ) {
                    Text("OK", color = Color.White)
                }
            },
            title = {
                Text("Verification Review Pending", fontWeight = FontWeight.Bold, color = Ink)
            },
            text = {
                Text("Your professional studio setup is currently being reviewed by an administrator. Please wait for approval before covering events.", color = Ink)
            },
            containerColor = BoneDeep
        )
    }

    if (showIncompleteAlert) {
        AlertDialog(
            onDismissRequest = { showIncompleteAlert = false },
            confirmButton = {
                Button(
                    onClick = { showIncompleteAlert = false },
                    colors = ButtonDefaults.buttonColors(containerColor = Fresh)
                ) {
                    Text("OK", color = Color.White)
                }
            },
            title = {
                Text("Onboarding Setup Required", fontWeight = FontWeight.Bold, color = Ink)
            },
            text = {
                Text("Your studio setup is not approved. Please complete the setup on the Settings tab and wait for administrator approval.", color = Ink)
            },
            containerColor = BoneDeep
        )
    }
}

@Composable
private fun EventPickerCard(
    event: PhotographerEventSummaryDto,
    onClick: () -> Unit,
) {
    val resolvedUrl = RetrofitClient.resolveImageUrl(event.bannerUrl)
    val lifecycle = deriveEventState(event.date)
    val isUpcoming = lifecycle == EventState.UPCOMING
    val isClosed = !canUploadToEvent(event.date) && !isUpcoming
    val isLive = canUploadToEvent(event.date)

    val stateLabel = when {
        isUpcoming -> "OPENS ON ${eventDateLabel(event.date)}"
        isClosed -> "CLOSED"
        else -> "LIVE"
    }
    val tone = when {
        isUpcoming -> StatusTone.Neutral
        isClosed -> StatusTone.Neutral
        else -> StatusTone.Approved
    }

    Surface(
        shape = QpCardShape,
        color = if (isLive) BoneDeep else Bone,
        border = BorderStroke(1.dp, Line),
        onClick = onClick,
        enabled = isLive,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(132.dp)
                    .background(Line)
            ) {
                if (resolvedUrl != null) {
                    AsyncImage(
                        model = resolvedUrl,
                        contentDescription = null,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                } else {
                    Kicker(
                        text = "Banner · soon",
                        color = SlateSoft,
                        modifier = Modifier.align(Alignment.Center)
                    )
                }
            }
            Column(modifier = Modifier.padding(16.dp)) {
                val cityLabel = extractCity(event.location)
                Kicker(
                    text = "${eventDateLabel(event.date)} · ${if (cityLabel.isNotBlank()) cityLabel else event.location}",
                    color = Slate
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = event.name,
                    color = Ink,
                    style = Typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                    lineHeight = 22.sp
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = event.location,
                    color = Slate,
                    style = Typography.bodyMedium
                )
                Spacer(modifier = Modifier.height(12.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    StatusChip(text = stateLabel, tone = tone)
                    when {
                        isUpcoming -> {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = Icons.Default.Lock,
                                    contentDescription = null,
                                    tint = SlateSoft,
                                    modifier = Modifier.size(14.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Kicker(text = "Opens on race day", color = SlateSoft)
                            }
                        }
                        isClosed -> {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = Icons.Default.Lock,
                                    contentDescription = null,
                                    tint = SlateSoft,
                                    modifier = Modifier.size(14.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Kicker(text = "Upload window closed", color = SlateSoft)
                            }
                        }
                        else -> {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Kicker(text = "Cover event", color = Ink)
                                Spacer(modifier = Modifier.width(4.dp))
                                Icon(
                                    imageVector = Icons.Default.ArrowForward,
                                    contentDescription = null,
                                    tint = Ink,
                                    modifier = Modifier.size(14.dp)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

