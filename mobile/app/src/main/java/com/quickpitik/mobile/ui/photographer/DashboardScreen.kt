package com.quickpitik.mobile.ui.photographer

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
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
import com.quickpitik.mobile.ui.theme.*


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotographerDashboardScreen(
    viewModel: PhotographerDashboardViewModel,
    onLogout: () -> Unit
) {
    var currentTab by remember { mutableStateOf(0) } // 0 = Overview, 1 = Tether, 2 = Events, 3 = Earnings, 4 = Settings
    var shareEvent by remember { mutableStateOf<PhotographerEventSummaryDto?>(null) }
    var showProfilePreview by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        viewModel.fetchVerificationStatus()
        viewModel.fetchEvents()
        viewModel.fetchEarningsAndTransactions()
        viewModel.fetchMessages()
    }

    val verificationState by viewModel.verificationState.collectAsState()
    val showSettingsBadge = when (val state = verificationState) {
        is VerificationUiState.Success -> state.verification.status.lowercase() != "approved"
        else -> true
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
        Scaffold(
            bottomBar = {
                NavigationBar(
                    containerColor = BoneDeep,
                    contentColor = Fresh,
                    modifier = Modifier.height(72.dp)
                ) {
                    NavigationBarItem(
                        selected = currentTab == 0,
                        onClick = { 
                            currentTab = 0 
                            viewModel.fetchVerificationStatus()
                            viewModel.fetchEvents()
                            viewModel.fetchEarningsAndTransactions()
                            viewModel.fetchMessages()
                            viewModel.fetchSettings()
                        },
                        icon = { Icon(Icons.Default.Info, contentDescription = "Overview") },
                        label = { Text("Overview", fontSize = 11.sp) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = Fresh,
                            selectedTextColor = Fresh,
                            unselectedIconColor = SlateSoft,
                            unselectedTextColor = SlateSoft,
                            indicatorColor = Bone
                        )
                    )
                    NavigationBarItem(
                        selected = currentTab == 1,
                        onClick = { currentTab = 1 },
                        icon = { Icon(Icons.Default.AddCircle, contentDescription = "Upload") },
                        label = { Text("Upload", fontSize = 11.sp) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = Fresh,
                            selectedTextColor = Fresh,
                            unselectedIconColor = SlateSoft,
                            unselectedTextColor = SlateSoft,
                            indicatorColor = Bone
                        )
                    )
                    NavigationBarItem(
                        selected = currentTab == 2,
                        onClick = { 
                            currentTab = 2
                            viewModel.fetchEvents()
                        },
                        icon = { Icon(Icons.Default.List, contentDescription = "Events") },
                        label = { Text("Events", fontSize = 11.sp) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = Fresh,
                            selectedTextColor = Fresh,
                            unselectedIconColor = SlateSoft,
                            unselectedTextColor = SlateSoft,
                            indicatorColor = Bone
                        )
                    )
                    NavigationBarItem(
                        selected = currentTab == 3,
                        onClick = { 
                            currentTab = 3
                            viewModel.fetchEarningsAndTransactions()
                        },
                        icon = { Icon(Icons.Default.ShoppingCart, contentDescription = "Earnings") },
                        label = { Text("Earnings", fontSize = 11.sp) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = Fresh,
                            selectedTextColor = Fresh,
                            unselectedIconColor = SlateSoft,
                            unselectedTextColor = SlateSoft,
                            indicatorColor = Bone
                        )
                    )
                    NavigationBarItem(
                        selected = currentTab == 4,
                        onClick = { 
                            currentTab = 4
                            viewModel.fetchVerificationStatus()
                        },
                        icon = {
                            BadgedBox(
                                badge = {
                                    if (showSettingsBadge) {
                                        Badge(
                                            containerColor = Fresh,
                                            modifier = Modifier.size(6.dp)
                                        )
                                    }
                                }
                            ) {
                                Icon(Icons.Default.Settings, contentDescription = "Settings")
                            }
                        },
                        label = { Text("Settings", fontSize = 11.sp) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = Fresh,
                            selectedTextColor = Fresh,
                            unselectedIconColor = SlateSoft,
                            unselectedTextColor = SlateSoft,
                            indicatorColor = Bone
                        )
                    )
                }
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
                        onPreviewProfile = { showProfilePreview = true }
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
                    2 -> PhotographerEventsScreen(viewModel = viewModel, onOpenShare = { shareEvent = it })
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
    val eventsState by viewModel.eventsState.collectAsState()
    val queueStats by viewModel.queueStats.collectAsState()
    var showDropdown by remember { mutableStateOf(false) }
    val scrollState = rememberScrollState()
    val context = LocalContext.current

    val pickMultipleMedia = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickMultipleVisualMedia(maxItems = 100)
    ) { uris ->
        if (uris.isNotEmpty()) {
            viewModel.queuePhotosFromGallery(context, uris)
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Bone)
            .padding(18.dp)
            .verticalScroll(scrollState)
            .statusBarsPadding()
            .navigationBarsPadding(),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // 0. Back button to clear active event and go to the picker list
        Row(
            modifier = Modifier
                .clickable { viewModel.selectEvent(null) }
                .padding(vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.ArrowBack,
                contentDescription = "Back",
                tint = Fresh,
                modifier = Modifier.size(16.dp)
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = "Back to event picker",
                color = Fresh,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold
            )
        }

        // 1. Premium Header
        Column(
            modifier = Modifier.fillMaxWidth().padding(bottom = 4.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .background(Fresh.copy(alpha = 0.1f), RoundedCornerShape(6.dp))
                    .padding(horizontal = 8.dp, vertical = 3.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .clip(CircleShape)
                        .background(Fresh)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "TETHER STREAM ACTIVE",
                    color = Fresh,
                    fontSize = 9.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 0.5.sp
                )
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "DSLR Tether Console",
                color = Ink,
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold
            )
        }


        // 2. Active Event Selector Card
        Card(
            onClick = { showDropdown = true },
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = BoneDeep),
            border = BorderStroke(1.dp, Line),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(18.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Place,
                            contentDescription = "Event location",
                            tint = Fresh,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "ACTIVE EVENT",
                            color = Fresh,
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        )
                    }
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .background(Line, RoundedCornerShape(4.dp))
                            .padding(horizontal = 6.dp, vertical = 2.dp)
                    ) {
                        Text(
                            text = "CHANGE ▾",
                            color = SlateSoft,
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = activeEvent?.name ?: "No active tether event selected",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Ink
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = activeEvent?.let { "${it.location} • ${it.state.uppercase()}" } ?: "Select an event from the Events tab or tap Change",
                    fontSize = 12.sp,
                    color = SlateSoft
                )

                DropdownMenu(
                    expanded = showDropdown,
                    onDismissRequest = { showDropdown = false },
                    modifier = Modifier.background(BoneDeep)
                ) {
                    when (val state = eventsState) {
                        is EventsState.Loading -> {
                            DropdownMenuItem(
                                text = { Text("Loading assigned events...", color = Ink, fontSize = 13.sp) },
                                onClick = {}
                            )
                        }
                        is EventsState.Success -> {
                            if (state.events.isEmpty()) {
                                DropdownMenuItem(
                                    text = { Text("No events found", color = Ink, fontSize = 13.sp) },
                                    onClick = {}
                                )
                            } else {
                                state.events.forEach { event ->
                                    DropdownMenuItem(
                                        text = { Text(event.name, color = Ink, fontSize = 13.sp) },
                                        onClick = {
                                            viewModel.selectEvent(event)
                                            showDropdown = false
                                        }
                                    )
                                }
                            }
                        }
                        is EventsState.Error -> {
                            DropdownMenuItem(
                                text = { Text("Error: ${state.message}", color = ErrorRed, fontSize = 13.sp) },
                                onClick = {}
                            )
                        }
                    }
                }
            }
        }

        // 3. Connection Status Card
        Card(
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = BoneDeep),
            border = BorderStroke(1.dp, Line),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(18.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Info,
                            contentDescription = "Camera",
                            tint = SlateSoft,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "CAMERA CONNECTION STATUS",
                            color = SlateSoft,
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        )
                    }
                    Box(
                        modifier = Modifier
                            .size(10.dp)
                            .clip(CircleShape)
                            .background(Fresh)
                    )
                }
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = "USB Camera Connected",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = Ink
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = "Sony Alpha 7 IV • CCID 4429",
                    fontSize = 13.sp,
                    color = SlateSoft,
                    fontWeight = FontWeight.Medium
                )
            }
        }

        // 4. Simulated SNAPSHOT trigger card
        Card(
            onClick = { viewModel.simulatePhotoCapture() },
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = BoneDeep),
            border = BorderStroke(1.5.dp, Fresh.copy(alpha = 0.6f)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier
                    .padding(18.dp)
                    .fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Add,
                        contentDescription = "Snapshot",
                        tint = Fresh,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "SIMULATE DSLR SNAPSHOT",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = Fresh,
                        letterSpacing = 0.5.sp
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Simulates a new photo capture over OTG connection",
                    fontSize = 11.sp,
                    color = SlateSoft,
                    textAlign = TextAlign.Center
                )
            }
        }

        // 4b. MANUAL GALLERY UPLOAD card
        Card(
            onClick = {
                pickMultipleMedia.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
            },
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = BoneDeep),
            border = BorderStroke(1.5.dp, Fresh.copy(alpha = 0.6f)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier
                    .padding(18.dp)
                    .fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.AddCircle,
                        contentDescription = "Gallery",
                        tint = Fresh,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "CHOOSE PHOTOS FROM GALLERY",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = Fresh,
                        letterSpacing = 0.5.sp
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Select one or more photos from your local device to upload to this event",
                    fontSize = 11.sp,
                    color = SlateSoft,
                    textAlign = TextAlign.Center
                )
            }
        }

        // 5. Sync Queue Stats Card
        Card(
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = BoneDeep),
            border = BorderStroke(1.dp, Line),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(18.dp)) {
                Text(
                    text = "UPLOAD SYNC ENGINE",
                    color = SlateSoft,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp
                )
                Spacer(modifier = Modifier.height(14.dp))

                // Live Progress bar
                LinearProgressIndicator(
                    progress = queueStats.progress,
                    color = Fresh,
                    trackColor = Line,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp))
                )
                Spacer(modifier = Modifier.height(12.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Synced: ${queueStats.syncedCount} photos",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        color = Ink
                    )
                    
                    val statusText = when {
                        queueStats.uploadingCount > 0 -> "Uploading..."
                        queueStats.queuedCount > 0 -> "Queued: ${queueStats.queuedCount}"
                        queueStats.failedCount > 0 -> "Failed: ${queueStats.failedCount}"
                        else -> "Idle"
                    }
                    val statusColor = if (queueStats.queuedCount > 0 || queueStats.uploadingCount > 0) Fresh else SlateSoft
                    
                    Text(
                        text = statusText,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = statusColor,
                        modifier = Modifier
                            .background(statusColor.copy(alpha = 0.1f), RoundedCornerShape(4.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
                
                Spacer(modifier = Modifier.height(18.dp))

                // Trigger sync engine action button
                Button(
                    onClick = { viewModel.runSyncEngine() },
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Fresh,
                        contentColor = Color.White
                    ),
                    modifier = Modifier.fillMaxWidth().height(46.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = "Sync",
                            tint = Color.White,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = if (queueStats.uploadingCount > 0) "SYNCING DSLR QUEUE..." else "RUN TETHER SYNC ENGINE",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
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
            .padding(18.dp)
            .statusBarsPadding()
            .navigationBarsPadding(),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .background(Fresh.copy(alpha = 0.1f), RoundedCornerShape(6.dp))
                    .padding(horizontal = 8.dp, vertical = 3.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .clip(CircleShape)
                        .background(Fresh)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "SELECT AN EVENT TO START UPLOADING",
                    color = Fresh,
                    fontSize = 9.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 0.5.sp
                )
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Upload Pictures",
                color = Ink,
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Select one of the public marathons or platform events below that you are not actively covering to start uploading photos.",
                color = SlateSoft,
                fontSize = 12.sp,
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
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = state.message,
                            color = MaterialTheme.colorScheme.error,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.padding(16.dp)
                        )
                        Button(
                            onClick = { viewModel.fetchPublicEvents() },
                            colors = ButtonDefaults.buttonColors(containerColor = Fresh)
                        ) {
                            Text("Retry", color = Color.White)
                        }
                    }
                }
            }
            is EventsState.Success -> {
                val notJoinedEvents = state.events.filter { pub ->
                    assignedEvents.none { ass -> ass.id == pub.id }
                }

                if (notJoinedEvents.isEmpty()) {
                    Box(
                        modifier = Modifier.fillMaxWidth().weight(1f).padding(24.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "No new platform events to join. You are actively covering all available events!",
                            color = SlateSoft,
                            fontSize = 13.sp,
                            textAlign = TextAlign.Center
                        )
                    }
                } else {
                    LazyColumn(
                        modifier = Modifier.fillMaxWidth().weight(1f),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        items(notJoinedEvents) { event ->
                            Card(
                                shape = RoundedCornerShape(16.dp),
                                colors = CardDefaults.cardColors(containerColor = BoneDeep),
                                border = BorderStroke(1.dp, Line),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Column {
                                    val resolvedUrl = event.bannerUrl?.let { url ->
                                        if (url.startsWith("/")) "http://10.0.2.2:8080$url"
                                        else url.replace("localhost", "10.0.2.2").replace("127.0.0.1", "10.0.2.2")
                                    }
                                    Box(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .height(130.dp)
                                            .background(Line)
                                    ) {
                                        if (resolvedUrl != null) {
                                            AsyncImage(
                                                model = resolvedUrl,
                                                contentDescription = "Event banner",
                                                contentScale = ContentScale.Crop,
                                                modifier = Modifier.fillMaxSize()
                                            )
                                        } else {
                                            Text(
                                                "BANNER · SOON",
                                                fontSize = 11.sp,
                                                fontWeight = FontWeight.Bold,
                                                color = Ink.copy(alpha = 0.3f),
                                                modifier = Modifier.align(Alignment.Center)
                                            )
                                        }

                                        // Status Chip at top-left of image
                                        val badgeColor = when (event.state.lowercase()) {
                                            "live", "open" -> Fresh
                                            "upcoming" -> MaterialTheme.colorScheme.primary
                                            else -> SlateSoft
                                        }
                                        Box(
                                            modifier = Modifier
                                                .align(Alignment.TopStart)
                                                .padding(10.dp)
                                                .clip(RoundedCornerShape(percent = 100))
                                                .background(Ink.copy(alpha = 0.55f))
                                                .padding(horizontal = 10.dp, vertical = 5.dp)
                                        ) {
                                            Row(
                                                verticalAlignment = Alignment.CenterVertically,
                                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                                            ) {
                                                Box(
                                                    modifier = Modifier
                                                        .size(6.dp)
                                                        .clip(CircleShape)
                                                        .background(badgeColor)
                                                )
                                                Text(
                                                    text = if (event.state.lowercase() == "live" || event.state.lowercase() == "open") "PHOTOS UPLOADING" else event.state.uppercase(),
                                                    color = badgeColor,
                                                    fontSize = 9.sp,
                                                    fontWeight = FontWeight.Bold,
                                                    letterSpacing = 0.8.sp
                                                )
                                            }
                                        }
                                    }

                                    Column(modifier = Modifier.padding(14.dp)) {
                                        // Header kicker matching Buyer card: Date · CITY (uppercase)
                                        val cityLabel = extractCity(event.location).uppercase()
                                        Text(
                                            text = "${event.date} · ${if (cityLabel.isNotBlank()) cityLabel else "CEBU"}",
                                            color = Slate,
                                            fontSize = 11.sp,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Spacer(modifier = Modifier.height(4.dp))

                                        Text(
                                            text = event.name,
                                            color = Ink,
                                            fontSize = 14.sp,
                                            fontWeight = FontWeight.Bold
                                        )

                                        Spacer(modifier = Modifier.height(2.dp))

                                        Text(
                                            text = event.location,
                                            color = SlateSoft,
                                            fontSize = 11.sp
                                        )

                                        Spacer(modifier = Modifier.height(12.dp))

                                        Button(
                                            onClick = {
                                                if (currentStatus == "approved") {
                                                    onSelectEvent(event)
                                                } else if (isRejected) {
                                                    showIncompleteAlert = true
                                                } else if (currentStatus == "pending") {
                                                    showPendingAlert = true
                                                } else {
                                                    showIncompleteAlert = true
                                                }
                                            },
                                            colors = ButtonDefaults.buttonColors(
                                                containerColor = Fresh.copy(alpha = 0.15f),
                                                contentColor = Fresh
                                            ),
                                            shape = RoundedCornerShape(8.dp),
                                            modifier = Modifier.fillMaxWidth().height(36.dp),
                                            contentPadding = PaddingValues(0.dp)
                                        ) {
                                            Text(
                                                text = "UPLOAD PHOTOS",
                                                fontWeight = FontWeight.Bold,
                                                fontSize = 11.sp
                                            )
                                        }
                                    }
                                }
                            }
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
