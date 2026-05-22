package com.quickpitik.mobile.ui.photographer

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.draw.clip
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
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotographerDashboardScreen(
    viewModel: PhotographerDashboardViewModel,
    onLogout: () -> Unit
) {
    var currentTab by remember { mutableStateOf(0) } // 0 = Overview, 1 = Tether, 2 = Events, 3 = Earnings, 4 = Settings

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
        )
    ) {
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
                        icon = { Icon(Icons.Default.Face, contentDescription = "Tether") },
                        label = { Text("Tether", fontSize = 11.sp) },
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
                        }
                    )
                    1 -> TetherConsoleView(viewModel = viewModel, onLogout = onLogout)
                    2 -> PhotographerEventsScreen(viewModel = viewModel)
                    3 -> PhotographerEarningsScreen(viewModel = viewModel)
                    4 -> PhotographerSettingsScreen(viewModel = viewModel)
                }
            }
        }
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TetherConsoleView(
    viewModel: PhotographerDashboardViewModel,
    onLogout: () -> Unit,
    modifier: Modifier = Modifier
) {
    val activeEvent by viewModel.activeEvent.collectAsState()
    val eventsState by viewModel.eventsState.collectAsState()
    val queueStats by viewModel.queueStats.collectAsState()
    var showDropdown by remember { mutableStateOf(false) }
    val scrollState = rememberScrollState()

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
        // 1. Premium Header Row
        Row(
            modifier = Modifier.fillMaxWidth().padding(bottom = 4.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
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
            
            Button(
                onClick = onLogout,
                colors = ButtonDefaults.buttonColors(
                    containerColor = BoneDeep,
                    contentColor = Ink
                ),
                border = BorderStroke(1.dp, Line),
                shape = RoundedCornerShape(12.dp),
                contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp),
                modifier = Modifier.height(36.dp)
            ) {
                Text("LEAVE", fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
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
