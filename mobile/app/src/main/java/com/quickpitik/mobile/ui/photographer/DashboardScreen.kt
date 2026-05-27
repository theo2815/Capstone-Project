package com.quickpitik.mobile.ui.photographer

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*

import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.data.remote.PhotographerMessageDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.usb.CameraConnectionState
import com.quickpitik.mobile.ui.runner.canUploadToEvent
import com.quickpitik.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotographerDashboardScreen(
    viewModel: PhotographerDashboardViewModel,
    onLogout: () -> Unit
) {
    var currentTab by remember { mutableStateOf(0) } // 0 = Home, 1 = Capture, 2 = Events, 3 = Earnings, 4 = Settings
    var shareEvent by remember { mutableStateOf<PhotographerEventSummaryDto?>(null) }
    var showProfilePreview by remember { mutableStateOf(false) }
    var showNotifDialog by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        viewModel.fetchVerificationStatus()
        viewModel.fetchEvents()
        viewModel.fetchEarningsAndTransactions()
        viewModel.fetchMessages()
        viewModel.fetchSettings()
    }

    val verificationState by viewModel.verificationState.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val brandSettings by viewModel.brandSettings.collectAsState()
    val showSettingsBadge = when (val state = verificationState) {
        is VerificationUiState.Success -> state.verification.status.lowercase() != "approved"
        else -> true
    }
    val unreadCount = remember(messages) { messages.count { it.readAt == null } }
    val resolvedAvatarUrl = brandSettings?.avatarUrl?.let {
        // M-2 (2026-05-27 PM): derive host from RetrofitClient.BASE_URL so
        // physical-device Wi-Fi setups pick up the right host without an edit here.
        it.replace("localhost", RetrofitClient.backendHost)
            .replace("127.0.0.1", RetrofitClient.backendHost)
    }

    // Explicitly lock the Photographer Dashboard into the premium athletic warm cream theme
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Fresh,
            onPrimary = Color.White,
            background = Bone,
            onBackground = Ink,
            surface = BoneDeep,
            onSurface = Ink,
            outline = Line
        ),
        // Carry the Quiet Studio type scale into the nested theme — otherwise the
        // whole photographer dashboard reverts to the system font (Material default).
        typography = Typography
    ) {
        val activeShareEvent = shareEvent
        if (activeShareEvent != null) {
            PhotographerEventShareScreen(
                event = activeShareEvent,
                viewModel = viewModel,
                onBack = { shareEvent = null }
            )
            return@MaterialTheme
        }
        if (showProfilePreview) {
            PhotographerPublicProfileScreen(
                viewModel = viewModel,
                onBack = { showProfilePreview = false }
            )
            return@MaterialTheme
        }
        if (showNotifDialog) {
            NotificationsInboxDialog(
                messages = messages,
                onDismiss = { showNotifDialog = false },
                onMarkRead = { id -> viewModel.markMessageAsRead(id) },
                onMarkAllRead = { viewModel.markAllMessagesAsRead() },
                onRemove = { id -> viewModel.removeMessage(id) },
            )
        }
        Scaffold(
            topBar = {
                PhotographerTopBar(
                    avatarUrl = resolvedAvatarUrl,
                    unreadCount = unreadCount,
                    onOpenNotifications = { showNotifDialog = true },
                    onPreviewProfile = { showProfilePreview = true },
                    onLogout = onLogout,
                )
            },
            bottomBar = {
                PhotographerFloatingBottomNav(
                    currentTab = currentTab,
                    showSettingsBadge = showSettingsBadge,
                    onTabSelected = { tab ->
                        currentTab = tab
                        when (tab) {
                            0 -> {
                                viewModel.fetchVerificationStatus()
                                viewModel.fetchEvents()
                                viewModel.fetchEarningsAndTransactions()
                                viewModel.fetchMessages()
                                viewModel.fetchSettings()
                            }
                            2 -> viewModel.fetchEvents()
                            3 -> viewModel.fetchEarningsAndTransactions()
                            4 -> viewModel.fetchVerificationStatus()
                        }
                    },
                )
            }
        ) { paddingValues ->
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Bone)
                    .padding(paddingValues)
            ) {
                when (currentTab) {
                    0 -> PhotographerOverviewScreen(
                        viewModel = viewModel,
                        onNavigateToSettings = {
                            currentTab = 4
                            viewModel.fetchVerificationStatus()
                        },
                        onNavigateToTab = { tab ->
                            currentTab = tab
                            when (tab) {
                                0 -> {
                                    viewModel.fetchVerificationStatus()
                                    viewModel.fetchEvents()
                                    viewModel.fetchEarningsAndTransactions()
                                    viewModel.fetchMessages()
                                    viewModel.fetchSettings()
                                }
                                2 -> viewModel.fetchEvents()
                                3 -> viewModel.fetchEarningsAndTransactions()
                                4 -> viewModel.fetchVerificationStatus()
                            }
                        },
                    )
                    1 -> {
                        val activeEvent by viewModel.activeEvent.collectAsState()
                        if (activeEvent == null) {
                            PublicEventPickerList(
                                viewModel = viewModel,
                                onSelectEvent = { event ->
                                    viewModel.selectEvent(event)
                                }
                            )
                        } else {
                            TetherConsoleView(viewModel = viewModel)
                        }
                    }
                    2 -> PhotographerEventsScreen(
                        viewModel = viewModel,
                        onOpenShare = { shareEvent = it },
                        onSyncEvent = { ev ->
                            viewModel.selectEvent(ev)
                            currentTab = 1
                        },
                    )
                    3 -> PhotographerEarningsScreen(viewModel = viewModel)
                    4 -> PhotographerSettingsScreen(viewModel = viewModel, onLogout = onLogout)
                }
            }
        }
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
    val scrollState = rememberScrollState()

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
                "UPCOMING" -> StatusTone.Warning
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
            is CameraConnectionState.Connected -> CameraConnectedCard(
                deviceName = cam.deviceName,
                vendorId = cam.vendorId,
                productId = cam.productId,
                onSimulate = { viewModel.simulatePhotoCapture() }
            )
            CameraConnectionState.Searching,
            CameraConnectionState.Disconnected -> CameraConnectPrompt(
                isSearching = cam is CameraConnectionState.Searching,
                onRescan = { viewModel.refreshCameraConnection() }
            )
        }

        // 3. Sync queue — only visible once a camera is actually tethered.
        // No connection ⇒ nothing to sync, so the card is hidden entirely.
        if (cameraState is CameraConnectionState.Connected) {
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
            Spacer(modifier = Modifier.height(16.dp))
            GhostCta(
                text = if (queueStats.uploadingCount > 0) "Syncing…" else "Run sync engine",
                onClick = { viewModel.runSyncEngine() },
                modifier = Modifier.fillMaxWidth(),
                enabled = queueStats.uploadingCount == 0
            )
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
    onSimulate: () -> Unit,
) {
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
        // Until the real tether stream lands, simulate stands in as the
        // primary path that exercises the camera → cloud → gallery pipeline.
        PrimaryCta(
            text = "Simulate DSLR shoot",
            onClick = onSimulate,
            modifier = Modifier.fillMaxWidth()
        )
    }
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
                // Picker contract: show every event that's currently within the
                // race-day + 3-day upload window — including events the
                // photographer is ALREADY covering (so they can return to dump
                // more frames mid-event). Mirrors the website's
                // canUploadToEvent gate; backend enforces the same window
                // server-side as EVENT_NOT_UPLOADABLE.
                val uploadableEvents = (state.events + assignedEvents)
                    .distinctBy { it.id }
                    .filter { canUploadToEvent(it.date) }

                if (uploadableEvents.isEmpty()) {
                    Box(
                        modifier = Modifier.fillMaxWidth().weight(1f).padding(24.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Kicker(text = "Nothing open for upload", color = Slate)
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "Capture opens on race day and stays open for 3 days after each event.",
                                color = Slate,
                                style = Typography.bodyMedium,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                } else {
                    LazyColumn(
                        modifier = Modifier.fillMaxWidth().weight(1f),
                        verticalArrangement = Arrangement.spacedBy(14.dp)
                    ) {
                        items(uploadableEvents) { event ->
                            EventPickerCard(
                                event = event,
                                onClick = {
                                    when {
                                        currentStatus == "approved" -> onSelectEvent(event)
                                        isRejected -> showIncompleteAlert = true
                                        currentStatus == "pending" -> showPendingAlert = true
                                        else -> showIncompleteAlert = true
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

private fun extractCity(location: String): String {
    val idx = location.lastIndexOf(',')
    return if (idx == -1) location.trim() else location.substring(idx + 1).trim()
}

@Composable
private fun EventPickerCard(
    event: PhotographerEventSummaryDto,
    onClick: () -> Unit,
) {
    val resolvedUrl = event.bannerUrl?.let { url ->
        // M-2 (2026-05-27 PM): host derived from RetrofitClient.BASE_URL.
        if (url.startsWith("/")) "${RetrofitClient.backendOrigin}$url"
        else url.replace("localhost", RetrofitClient.backendHost)
            .replace("127.0.0.1", RetrofitClient.backendHost)
    }
    val stateLabel = event.state.uppercase()
    val tone = when (stateLabel) {
        "LIVE", "OPEN" -> StatusTone.Approved
        "UPCOMING" -> StatusTone.Warning
        "CLOSED", "ENDED" -> StatusTone.Neutral
        else -> StatusTone.Neutral
    }
    Surface(
        shape = QpCardShape,
        color = BoneDeep,
        border = BorderStroke(1.dp, Line),
        onClick = onClick,
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
                    text = "${event.date} · ${if (cityLabel.isNotBlank()) cityLabel else event.location}",
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

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PhotographerTopBar(
    avatarUrl: String?,
    unreadCount: Int,
    onOpenNotifications: () -> Unit,
    onPreviewProfile: () -> Unit,
    onLogout: () -> Unit,
) {
    var showAvatarMenu by remember { mutableStateOf(false) }
    TopAppBar(
        title = {
            Column {
                Text(
                    text = "STUDIO",
                    style = Typography.labelMedium,
                    color = Slate,
                )
                Text(
                    text = "QuickPitik",
                    style = Typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = Ink,
                )
            }
        },
        actions = {
            IconButton(onClick = onOpenNotifications) {
                Box {
                    Icon(
                        imageVector = Icons.Default.Notifications,
                        contentDescription = "Notifications",
                        tint = if (unreadCount > 0) Fresh else SlateSoft,
                    )
                    if (unreadCount > 0) {
                        Box(
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .size(14.dp)
                                .clip(CircleShape)
                                .background(ErrorRed)
                                .border(2.dp, Bone, CircleShape),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(
                                text = if (unreadCount > 9) "9+" else unreadCount.toString(),
                                color = Color.White,
                                fontSize = 8.sp,
                                fontWeight = FontWeight.Bold,
                            )
                        }
                    }
                }
            }
            Box {
                IconButton(onClick = { showAvatarMenu = true }) {
                    Box(
                        modifier = Modifier
                            .size(32.dp)
                            .clip(CircleShape)
                            .background(BoneDeep)
                            .border(1.dp, Line, CircleShape),
                        contentAlignment = Alignment.Center,
                    ) {
                        if (!avatarUrl.isNullOrBlank()) {
                            AsyncImage(
                                model = avatarUrl,
                                contentDescription = "Profile menu",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize().clip(CircleShape),
                            )
                        } else {
                            Icon(
                                imageVector = Icons.Default.Person,
                                contentDescription = "Profile menu",
                                tint = SlateSoft,
                                modifier = Modifier.size(20.dp),
                            )
                        }
                    }
                }
                DropdownMenu(
                    expanded = showAvatarMenu,
                    onDismissRequest = { showAvatarMenu = false },
                    modifier = Modifier.background(BoneDeep),
                ) {
                    DropdownMenuItem(
                        text = { Text("Preview public profile", color = Ink, fontSize = 13.sp) },
                        leadingIcon = {
                            Icon(
                                imageVector = Icons.Default.Person,
                                contentDescription = null,
                                tint = Ink,
                                modifier = Modifier.size(18.dp),
                            )
                        },
                        onClick = {
                            showAvatarMenu = false
                            onPreviewProfile()
                        },
                    )
                    DropdownMenuItem(
                        text = { Text("Switch to runner", color = SlateSoft, fontSize = 13.sp) },
                        trailingIcon = {
                            Text(
                                text = "SOON",
                                color = SlateSoft,
                                fontSize = 9.sp,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 0.8.sp,
                            )
                        },
                        enabled = false,
                        onClick = {},
                    )
                    Divider(color = Line)
                    DropdownMenuItem(
                        text = { Text("Sign out", color = ErrorRed, fontSize = 13.sp) },
                        leadingIcon = {
                            Icon(
                                imageVector = Icons.Default.ExitToApp,
                                contentDescription = null,
                                tint = ErrorRed,
                                modifier = Modifier.size(18.dp),
                            )
                        },
                        onClick = {
                            showAvatarMenu = false
                            onLogout()
                        },
                    )
                }
            }
            Spacer(modifier = Modifier.width(4.dp))
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = Bone,
            scrolledContainerColor = Bone,
            navigationIconContentColor = Ink,
            titleContentColor = Ink,
            actionIconContentColor = Ink,
        ),
    )
}

@Composable
private fun NotificationsInboxDialog(
    messages: List<PhotographerMessageDto>,
    onDismiss: () -> Unit,
    onMarkRead: (String) -> Unit,
    onMarkAllRead: () -> Unit,
    onRemove: (String) -> Unit,
) {
    var removeTarget by remember { mutableStateOf<PhotographerMessageDto?>(null) }
    val unreadCount = messages.count { it.readAt == null }

    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            OutlinedButton(
                onClick = onDismiss,
                shape = PillShape,
                border = BorderStroke(1.dp, Ink),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink),
                modifier = Modifier.height(40.dp),
            ) {
                Text(
                    text = "CLOSE",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.5.sp,
                )
            }
        },
        title = {
            Text(
                text = "Inbox",
                color = Ink,
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Column(modifier = Modifier.fillMaxWidth()) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Kicker(
                        text = if (unreadCount > 0) "$unreadCount unread · ${messages.size} total"
                               else "${messages.size} total",
                        color = SlateSoft,
                    )
                    if (unreadCount > 0) {
                        Text(
                            text = "MARK ALL READ",
                            color = Slate,
                            style = Typography.labelMedium,
                            modifier = Modifier
                                .clickable { onMarkAllRead() }
                                .padding(horizontal = 4.dp, vertical = 4.dp),
                        )
                    }
                }
                if (messages.isEmpty()) {
                    Text(
                        text = "No messages yet. Admin actions on your account will land here.",
                        color = SlateSoft,
                        fontSize = 14.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 32.dp),
                    )
                } else {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 440.dp)
                            .verticalScroll(rememberScrollState()),
                    ) {
                        messages.forEachIndexed { index, msg ->
                            if (index > 0) Divider(color = Line)
                            InboxRow(
                                message = msg,
                                onMarkRead = { onMarkRead(msg.id) },
                                onRemove = { removeTarget = msg },
                            )
                        }
                    }
                }
            }
        },
        containerColor = Bone,
        shape = QpCardShape,
    )

    val target = removeTarget
    if (target != null) {
        AlertDialog(
            onDismissRequest = { removeTarget = null },
            title = {
                Text(
                    text = "Remove this notification?",
                    color = Ink,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                )
            },
            text = {
                Text(
                    text = "This message will be cleared from your inbox. Admin still has the underlying record on the decision log.",
                    color = Slate,
                    fontSize = 14.sp,
                    lineHeight = 20.sp,
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        onRemove(target.id)
                        removeTarget = null
                    },
                    shape = PillShape,
                    colors = ButtonDefaults.buttonColors(containerColor = ErrorRed),
                    modifier = Modifier.height(40.dp),
                ) {
                    Text(
                        text = "REMOVE",
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.5.sp,
                    )
                }
            },
            dismissButton = {
                OutlinedButton(
                    onClick = { removeTarget = null },
                    shape = PillShape,
                    border = BorderStroke(1.dp, Ink),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink),
                    modifier = Modifier.height(40.dp),
                ) {
                    Text(
                        text = "CANCEL",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.5.sp,
                    )
                }
            },
            containerColor = Bone,
            shape = QpCardShape,
        )
    }
}

@Composable
private fun InboxRow(
    message: PhotographerMessageDto,
    onMarkRead: () -> Unit,
    onRemove: () -> Unit,
) {
    val isUnread = message.readAt == null
    val toneColor = when (message.kind) {
        "verification_approved", "unsuspended", "dispute_resolved",
        "payout_approved", "payout_paid", "payout_report_resolved" -> Fresh
        "verification_rejected", "verification_reset",
        "dispute_escalated", "payout_held" -> WarningOrange
        "suspended" -> ErrorRed
        "admin_message" -> Ink
        else -> Slate
    }
    val title = message.title?.trim().takeUnless { it.isNullOrEmpty() }
        ?: messageKindLabel(message.kind)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                if (isUnread) BoneDeep.copy(alpha = 0.4f) else Color.Transparent,
            )
            .clickable { if (isUnread) onMarkRead() }
            .padding(horizontal = 8.dp, vertical = 14.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (isUnread) {
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .clip(CircleShape)
                            .background(Fresh),
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                }
                Text(
                    text = messageKindLabel(message.kind).uppercase(),
                    style = Typography.labelMedium,
                    color = toneColor,
                )
            }
            Text(
                text = formatInboxDate(message.createdAt),
                style = Typography.labelMedium,
                color = SlateSoft,
            )
        }
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = title,
            color = if (isUnread) Ink else InkSoft,
            fontSize = 16.sp,
            fontWeight = if (isUnread) FontWeight.SemiBold else FontWeight.Normal,
            lineHeight = 22.sp,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = message.body,
            color = Slate,
            fontSize = 14.sp,
            lineHeight = 20.sp,
        )
        Spacer(modifier = Modifier.height(10.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End,
        ) {
            Text(
                text = "REMOVE",
                color = SlateSoft,
                style = Typography.labelMedium,
                modifier = Modifier
                    .clickable { onRemove() }
                    .padding(horizontal = 6.dp, vertical = 4.dp),
            )
        }
    }
}

private fun messageKindLabel(kind: String): String = when (kind) {
    "verification_approved" -> "Verification approved"
    "verification_rejected" -> "Verification needs changes"
    "verification_reset" -> "Verification reset"
    "suspended" -> "Account suspended"
    "unsuspended" -> "Account reinstated"
    "force_edit" -> "Profile updated by admin"
    "dispute_resolved" -> "Refund resolved"
    "dispute_denied" -> "Refund denied"
    "dispute_escalated" -> "Refund escalated"
    "payout_approved" -> "Payout approved"
    "payout_held" -> "Payout held"
    "payout_paid" -> "Payout paid"
    "payout_report_acknowledged" -> "Report acknowledged"
    "payout_report_resolved" -> "Report resolved"
    "admin_message" -> "Message from admin"
    else -> kind.replace('_', ' ').replaceFirstChar { it.uppercase() }
}

private fun formatInboxDate(iso: String): String {
    return try {
        val parts = iso.substring(0, 10).split("-")
        val months = listOf(
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
            "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
        )
        "${months[parts[1].toInt() - 1]} ${parts[2].toInt()}"
    } catch (e: Exception) { iso }
}

// Floating-pill bottom nav — Quiet Studio'd port of a glassmorphic source design.
// Flat-with-borders: Bone fill, 1dp Line outline, no blur / no shadow / no Fresh
// (Fresh stays reserved for the screen's primary CTA — the Settings dot is the
// one tolerated micro-accent, matching the website's notification badges).
@Composable
private fun PhotographerFloatingBottomNav(
    currentTab: Int,
    showSettingsBadge: Boolean,
    onTabSelected: (Int) -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .windowInsetsPadding(WindowInsets.navigationBars)
            .padding(horizontal = 12.dp, vertical = 12.dp),
        contentAlignment = Alignment.Center,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(PillShape)
                .background(Bone)
                .border(BorderStroke(1.dp, Line), PillShape)
                .padding(6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            FloatingNavItem(
                icon = Icons.Default.Home,
                label = "Home",
                selected = currentTab == 0,
                onClick = { onTabSelected(0) },
                modifier = Modifier.weight(1f),
            )
            FloatingNavItem(
                icon = Icons.Default.AddCircle,
                label = "Capture",
                selected = currentTab == 1,
                onClick = { onTabSelected(1) },
                modifier = Modifier.weight(1f),
            )
            FloatingNavItem(
                icon = Icons.Default.List,
                label = "Events",
                selected = currentTab == 2,
                onClick = { onTabSelected(2) },
                modifier = Modifier.weight(1f),
            )
            FloatingNavItem(
                icon = Icons.Default.ShoppingCart,
                label = "Earnings",
                selected = currentTab == 3,
                onClick = { onTabSelected(3) },
                modifier = Modifier.weight(1f),
            )
            FloatingNavItem(
                icon = Icons.Default.Settings,
                label = "Settings",
                selected = currentTab == 4,
                onClick = { onTabSelected(4) },
                modifier = Modifier.weight(1f),
                badge = showSettingsBadge,
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun FloatingNavItem(
    icon: ImageVector,
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    badge: Boolean = false,
) {
    val bg by animateColorAsState(
        targetValue = if (selected) Ink else Color.Transparent,
        animationSpec = tween(180),
        label = "navItemBg",
    )
    val tint by animateColorAsState(
        targetValue = if (selected) Bone else Slate,
        animationSpec = tween(180),
        label = "navItemTint",
    )
    Column(
        modifier = modifier
            .heightIn(min = 56.dp)
            .clip(PillShape)
            .background(bg)
            .clickable(onClick = onClick)
            .padding(vertical = 8.dp, horizontal = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        if (badge) {
            BadgedBox(
                badge = {
                    Badge(
                        containerColor = Fresh,
                        modifier = Modifier.size(6.dp),
                    )
                },
            ) {
                Icon(icon, contentDescription = label, tint = tint, modifier = Modifier.size(22.dp))
            }
        } else {
            Icon(icon, contentDescription = label, tint = tint, modifier = Modifier.size(22.dp))
        }
        Spacer(Modifier.height(2.dp))
        Text(
            text = label,
            color = tint,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            maxLines = 1,
        )
    }
}
