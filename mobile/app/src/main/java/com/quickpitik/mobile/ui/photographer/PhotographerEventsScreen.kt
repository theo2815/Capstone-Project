package com.quickpitik.mobile.ui.photographer

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.ui.theme.*
import coil.compose.AsyncImage
import androidx.compose.ui.layout.ContentScale
import android.widget.Toast
import androidx.compose.ui.platform.LocalContext

@Composable
fun PhotographerEventsScreen(
    viewModel: PhotographerDashboardViewModel,
    modifier: Modifier = Modifier,
    onOpenShare: (PhotographerEventSummaryDto) -> Unit = {}
) {
    val eventsState by viewModel.eventsState.collectAsState()
    val publicEventsState by viewModel.publicEventsState.collectAsState()
    val verificationState by viewModel.verificationState.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val activeEvent by viewModel.activeEvent.collectAsState()
    val context = LocalContext.current
    var selectedTab by remember { mutableStateOf(0) } // 0 = Discover Platform, 1 = Covered (Live), 2 = Covered (Past)

    var showPendingAlert by remember { mutableStateOf(false) }
    var showIncompleteAlert by remember { mutableStateOf(false) }

    // Derive rejection from real inbox messages:
    val latestMessage = remember(messages) { messages.maxByOrNull { it.createdAt } }
    val currentStatus = (verificationState as? VerificationUiState.Success)?.verification?.status?.lowercase() ?: "incomplete"
    val isRejected = (currentStatus == "incomplete" || currentStatus == "rejected") && (latestMessage?.kind == "verification_rejected")

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Bone) // warm cream background
            .padding(16.dp)
    ) {
        // Header
        Text(
            text = "Event Schedule",
            color = Ink,
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        // Subtitle instructions
        Text(
            text = if (selectedTab == 0) 
                "Discover published marathons and events in the system. Select one to set as your active DSLR tether event."
            else 
                "Below are marathons and races you are assigned to shoot. Use the Tether console to synchronize DSLR captures.",
            color = SlateSoft,
            fontSize = 13.sp,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        // Tab Selector Row
        TabRow(
            selectedTabIndex = selectedTab,
            containerColor = Bone,
            contentColor = Fresh,
            indicator = { tabPositions ->
                TabRowDefaults.Indicator(
                    modifier = Modifier.tabIndicatorOffset(tabPositions[selectedTab]),
                    color = Fresh
                )
            },
            modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp)
        ) {
            Tab(
                selected = selectedTab == 0,
                onClick = { selectedTab = 0 },
                text = {
                    Text(
                        text = "Discover Platform",
                        fontWeight = if (selectedTab == 0) FontWeight.Bold else FontWeight.Normal,
                        fontSize = 12.sp
                    )
                },
                selectedContentColor = Fresh,
                unselectedContentColor = SlateSoft
            )
            Tab(
                selected = selectedTab == 1,
                onClick = { selectedTab = 1 },
                text = {
                    Text(
                        text = "Covered (Live)",
                        fontWeight = if (selectedTab == 1) FontWeight.Bold else FontWeight.Normal,
                        fontSize = 12.sp
                    )
                },
                selectedContentColor = Fresh,
                unselectedContentColor = SlateSoft
            )
            Tab(
                selected = selectedTab == 2,
                onClick = { selectedTab = 2 },
                text = {
                    Text(
                        text = "Covered (Past)",
                        fontWeight = if (selectedTab == 2) FontWeight.Bold else FontWeight.Normal,
                        fontSize = 12.sp
                    )
                },
                selectedContentColor = Fresh,
                unselectedContentColor = SlateSoft
            )
        }

        // Render List based on States
        val currentState = if (selectedTab == 0) publicEventsState else eventsState

        when (val state = currentState) {
            is EventsState.Loading -> {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(color = Fresh)
                }
            }
            is EventsState.Error -> {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = state.message,
                            color = MaterialTheme.colorScheme.error,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.padding(16.dp)
                        )
                        Button(
                            onClick = { 
                                if (selectedTab == 0) {
                                    viewModel.fetchPublicEvents()
                                } else {
                                    viewModel.fetchEvents()
                                }
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = Fresh)
                        ) {
                            Text("Retry", color = Color.White)
                        }
                    }
                }
            }
            is EventsState.Success -> {
                // Filter items
                val filteredList = if (selectedTab == 0) {
                    state.events
                } else {
                    val baseList = state.events.filter { event ->
                        val isLiveOrActive = event.state.lowercase() == "live" || event.state.lowercase() == "open" || event.state.lowercase() == "active"
                        if (selectedTab == 1) isLiveOrActive else !isLiveOrActive
                    }
                    if (selectedTab == 1 && activeEvent != null) {
                        if (baseList.none { it.id == activeEvent?.id }) {
                            listOf(activeEvent!!) + baseList
                        } else {
                            baseList
                        }
                    } else {
                        baseList
                    }
                }

                if (filteredList.isEmpty()) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(32.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = when (selectedTab) {
                                1 -> "No active or live events assigned right now."
                                2 -> "No past or upcoming events scheduled."
                                else -> "No published platform events available."
                            },
                            color = SlateSoft,
                            fontSize = 14.sp,
                            textAlign = TextAlign.Center
                        )
                    }
                } else {
                    LazyColumn(
                        modifier = Modifier.fillMaxWidth().weight(1f),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        items(filteredList) { event ->
                            EventCard(
                                event = event,
                                onOpenShare = { onOpenShare(event) },
                                onSelectEvent = {
                                    if (currentStatus == "approved") {
                                        viewModel.selectEvent(event)
                                        Toast.makeText(context, "Active event set to: ${event.name}", Toast.LENGTH_SHORT).show()
                                    } else if (isRejected) {
                                        showIncompleteAlert = true
                                    } else if (currentStatus == "pending") {
                                        showPendingAlert = true
                                    } else {
                                        showIncompleteAlert = true
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
fun EventCard(
    event: PhotographerEventSummaryDto,
    onSelectEvent: () -> Unit,
    onOpenShare: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val isLive = event.state.lowercase() == "live" || event.state.lowercase() == "open" || event.state.lowercase() == "active"
    val borderStrokeColor = if (isLive) Fresh else Line

    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        shape = RoundedCornerShape(12.dp),
        modifier = modifier
            .fillMaxWidth()
            .border(BorderStroke(1.dp, borderStrokeColor), RoundedCornerShape(12.dp))
    ) {
        Column {
            val resolvedUrl = resolveImageUrl(event.bannerUrl)
            if (resolvedUrl != null) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(130.dp)
                        .background(Line)
                ) {
                    AsyncImage(
                        model = resolvedUrl,
                        contentDescription = "Event banner",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                }
            }

            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                // Event State Badges & Title Row
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = event.name,
                        color = Ink,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.weight(1f)
                    )

                    // State indicator Badge
                    val badgeColor = when (event.state.lowercase()) {
                        "live", "open" -> Fresh
                        "upcoming" -> MaterialTheme.colorScheme.primary
                        else -> SlateSoft
                    }

                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(4.dp))
                            .background(badgeColor.copy(alpha = 0.15f))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = event.state.uppercase(),
                            color = badgeColor,
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                Spacer(modifier = Modifier.height(4.dp))

                // Location
                Text(
                    text = "${event.location} • ${event.date}",
                    color = SlateSoft,
                    fontSize = 12.sp
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Statistics Row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    StatItem(label = "Photos Shot", value = "${event.photoCount}")
                    StatItem(label = "Photos Sold", value = "${event.salesCount}")
                    StatItem(
                        label = "Revenue Kept", 
                        value = "₱%,.2f".format(event.revenueKept), 
                        valueColor = Fresh
                    )
                }

                // Quick set-active action button for TETHERING
                if (isLive) {
                    Spacer(modifier = Modifier.height(14.dp))
                    Button(
                        onClick = onSelectEvent,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Fresh.copy(alpha = 0.15f),
                            contentColor = Fresh
                        ),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth().height(36.dp),
                        contentPadding = PaddingValues(0.dp)
                    ) {
                        Text(
                            text = "SET AS ACTIVE TETHER EVENT",
                            fontWeight = FontWeight.Bold,
                            fontSize = 11.sp
                        )
                    }
                }

                // View & share the public gallery for any covered event
                if (event.photoCount > 0) {
                    Spacer(modifier = Modifier.height(10.dp))
                    OutlinedButton(
                        onClick = onOpenShare,
                        border = BorderStroke(1.dp, Ink),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth().height(36.dp),
                        contentPadding = PaddingValues(0.dp)
                    ) {
                        Text(
                            text = "VIEW & SHARE GALLERY",
                            fontWeight = FontWeight.Bold,
                            fontSize = 11.sp
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StatItem(
    label: String,
    value: String,
    valueColor: Color = Ink
) {
    Column {
        Text(
            text = label.uppercase(),
            color = SlateSoft,
            fontSize = 9.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 0.5.sp
        )
        Spacer(modifier = Modifier.height(2.dp))
        Text(
            text = value,
            color = valueColor,
            fontSize = 15.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

private fun resolveImageUrl(url: String?): String? {
    if (url == null || url.trim().isEmpty()) return null
    if (url.startsWith("/")) {
        return "http://10.0.2.2:8080$url"
    }
    return url.replace("localhost", "10.0.2.2").replace("127.0.0.1", "10.0.2.2")
}
