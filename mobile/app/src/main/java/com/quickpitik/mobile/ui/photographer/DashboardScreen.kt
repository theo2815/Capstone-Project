package com.quickpitik.mobile.ui.photographer

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.quickpitik.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotographerDashboardScreen(
    viewModel: PhotographerDashboardViewModel,
    onLogout: () -> Unit
) {
    val activeEvent by viewModel.activeEvent.collectAsState()
    val eventsState by viewModel.eventsState.collectAsState()
    val queueStats by viewModel.queueStats.collectAsState()

    var showDropdown by remember { mutableStateOf(false) }

    // Explicitly lock the Photographer Dashboard into the premium athletic Dark Theme
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Fresh,
            onPrimary = Ink,
            background = Ink,
            onBackground = Bone,
            surface = InkSoft,
            onSurface = Bone,
            outline = Slate
        )
    ) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = Ink
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp)
                    .statusBarsPadding()
                    .navigationBarsPadding()
            ) {
                // Header Row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "TETHER CONSOLE",
                            style = Typography.labelMedium,
                            color = SlateSoft
                        )
                        Text(
                            text = "QuickPitik",
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Fresh
                        )
                    }
                    Button(
                        onClick = onLogout,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = InkSoft,
                            contentColor = Bone
                        ),
                        shape = RoundedCornerShape(12.dp),
                        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
                    ) {
                        Text("LEAVE", style = Typography.labelMedium)
                    }
                }
                Spacer(modifier = Modifier.height(28.dp))

                // Active Event Selector Card
                Card(
                    onClick = { showDropdown = true },
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = InkSoft),
                    border = BorderStroke(1.dp, Slate),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(20.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "ACTIVE EVENT",
                                style = Typography.labelMedium,
                                color = Fresh
                            )
                            Text(
                                text = "CHANGE ▾",
                                style = Typography.labelMedium,
                                color = SlateSoft
                            )
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = activeEvent?.name ?: "No assigned event",
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Bone
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = activeEvent?.let { "${it.location} • ${it.state.uppercase()}" } ?: "Verify with administrator",
                            style = Typography.bodyMedium,
                            color = SlateSoft
                        )

                        DropdownMenu(
                            expanded = showDropdown,
                            onDismissRequest = { showDropdown = false },
                            modifier = Modifier.background(InkSoft)
                        ) {
                            when (val state = eventsState) {
                                is EventsState.Loading -> {
                                    DropdownMenuItem(
                                        text = { Text("Loading assigned events...", color = Bone) },
                                        onClick = {}
                                    )
                                }
                                is EventsState.Success -> {
                                    if (state.events.isEmpty()) {
                                        DropdownMenuItem(
                                            text = { Text("No events found", color = Bone) },
                                            onClick = {}
                                        )
                                    } else {
                                        state.events.forEach { event ->
                                            DropdownMenuItem(
                                                text = { Text(event.name, color = Bone) },
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
                                        text = { Text("Error: ${state.message}", color = Color.Red) },
                                        onClick = {}
                                    )
                                }
                            }
                        }
                    }
                }
                Spacer(modifier = Modifier.height(20.dp))

                // Connection Status Card
                Card(
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = InkSoft),
                    border = BorderStroke(1.dp, Slate),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(20.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "CAMERA STATUS",
                                style = Typography.labelMedium,
                                color = SlateSoft
                            )
                            Surface(
                                shape = RoundedCornerShape(percent = 100),
                                color = Fresh,
                                modifier = Modifier.size(8.dp)
                            ) {}
                        }
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(
                            text = "USB Camera Connected",
                            style = Typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = Bone
                        )
                        Text(
                            text = "Sony Alpha 7 IV · CCID 4429",
                            style = Typography.bodyMedium,
                            color = SlateSoft
                        )
                    }
                }
                Spacer(modifier = Modifier.height(20.dp))

                // Simulated SNAPSHOT trigger card
                Card(
                    onClick = { viewModel.simulatePhotoCapture() },
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = InkSoft),
                    border = BorderStroke(1.dp, Fresh), // Highlighted in brand fresh green!
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp).fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "⚡ SIMULATE DSLR SNAPSHOT",
                            style = Typography.labelLarge,
                            fontWeight = FontWeight.Bold,
                            color = Fresh
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Tap to simulate a new photo capture over USB OTG",
                            style = Typography.bodySmall,
                            color = SlateSoft
                        )
                    }
                }
                Spacer(modifier = Modifier.height(20.dp))

                // Sync Queue Stats
                Card(
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = InkSoft),
                    border = BorderStroke(1.dp, Slate),
                    modifier = Modifier.fillMaxWidth().weight(1f)
                ) {
                    Column(
                        modifier = Modifier.padding(20.dp),
                        verticalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column {
                            Text(
                                text = "UPLOAD PROGRESS",
                                style = Typography.labelMedium,
                                color = SlateSoft
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            
                            // Live Room Progress bar
                            LinearProgressIndicator(
                                progress = queueStats.progress,
                                color = Fresh,
                                trackColor = Ink,
                                modifier = Modifier.fillMaxWidth().height(8.dp)
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text("Synced: ${queueStats.syncedCount} photos", style = Typography.bodyMedium, color = Bone)
                                Text(
                                    text = when {
                                        queueStats.uploadingCount > 0 -> "Uploading..."
                                        queueStats.queuedCount > 0 -> "Queued: ${queueStats.queuedCount} remaining"
                                        queueStats.failedCount > 0 -> "Failed: ${queueStats.failedCount} uploads"
                                        else -> "Idle"
                                    },
                                    style = Typography.bodyMedium,
                                    color = if (queueStats.queuedCount > 0 || queueStats.uploadingCount > 0) Fresh else SlateSoft
                                )
                            }
                        }

                        // Trigger sync engine action button
                        Button(
                            onClick = { viewModel.runSyncEngine() },
                            shape = RoundedCornerShape(percent = 100),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Fresh,
                                contentColor = Ink
                            ),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = if (queueStats.uploadingCount > 0) "SYNCING QUEUE..." else "RUN SYNC ENGINE",
                                style = Typography.labelLarge
                            )
                        }
                    }
                }
            }
        }
    }
}
