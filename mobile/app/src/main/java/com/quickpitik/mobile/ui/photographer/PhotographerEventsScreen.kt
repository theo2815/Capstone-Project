package com.quickpitik.mobile.ui.photographer

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.shape.CircleShape
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

@Composable
fun PhotographerEventsScreen(
    viewModel: PhotographerDashboardViewModel,
    modifier: Modifier = Modifier,
    onOpenShare: (PhotographerEventSummaryDto) -> Unit = {}
) {
    val eventsState by viewModel.eventsState.collectAsState()
    val activeEvent by viewModel.activeEvent.collectAsState()
    var selectedTab by remember { mutableStateOf(0) } // 0 = Covered (Live), 1 = Covered (Past)

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
            text = "Below are marathons and races you are assigned to shoot. Use the Upload Pics tab to synchronize DSLR or gallery captures.",
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
                        text = "Covered (Live)",
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
                        text = "Covered (Past)",
                        fontWeight = if (selectedTab == 1) FontWeight.Bold else FontWeight.Normal,
                        fontSize = 12.sp
                    )
                },
                selectedContentColor = Fresh,
                unselectedContentColor = SlateSoft
            )
        }

        // Render List based on States
        val currentState = eventsState

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
                                viewModel.fetchEvents()
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
                val baseList = state.events.filter { event ->
                    val isLiveOrActive = event.state.lowercase() == "live" || event.state.lowercase() == "open" || event.state.lowercase() == "active"
                    if (selectedTab == 0) isLiveOrActive else !isLiveOrActive
                }
                val filteredList = if (selectedTab == 0 && activeEvent != null) {
                    if (baseList.none { it.id == activeEvent?.id }) {
                        listOf(activeEvent!!) + baseList
                    } else {
                        baseList
                    }
                } else {
                    baseList
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
                                0 -> "No active or live events assigned right now."
                                else -> "No past or upcoming events scheduled."
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
                                onOpenShare = { onOpenShare(event) }
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun EventCard(
    event: PhotographerEventSummaryDto,
    onOpenShare: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val isLive = event.state.lowercase() == "live" || event.state.lowercase() == "open" || event.state.lowercase() == "active"
    val borderStrokeColor = if (isLive) Fresh else Line

    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.dp, borderStrokeColor),
        modifier = modifier.fillMaxWidth()
    ) {
        Column {
            val resolvedUrl = resolveImageUrl(event.bannerUrl)
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

                // State indicator Badge inside the banner top-left (matches Buyer design)
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

            Column(
                modifier = Modifier.padding(14.dp)
            ) {
                // Header Kicker matching Buyer card: Date · CITY (uppercase)
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
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(2.dp))

                Text(
                    text = event.location,
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

private fun extractCity(location: String): String {
    val idx = location.lastIndexOf(',')
    return if (idx == -1) location.trim() else location.substring(idx + 1).trim()
}

