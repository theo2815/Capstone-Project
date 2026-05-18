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
    onLogout: () -> Unit
) {
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

                // Active Event Card
                Card(
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = InkSoft),
                    border = BorderStroke(1.dp, Slate),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(20.dp)) {
                        Text(
                            text = "ACTIVE EVENT",
                            style = Typography.labelMedium,
                            color = Fresh
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "Cebu Marathon 2026",
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Bone
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "CClex Waterfront Route &middot; 42K",
                            style = Typography.bodyMedium,
                            color = SlateSoft
                        )
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
                            text = "Sony Alpha 7 IV &middot; CCID 4429",
                            style = Typography.bodyMedium,
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
                            
                            // Progress bar
                            LinearProgressIndicator(
                                progress = 0.72f,
                                color = Fresh,
                                trackColor = Ink,
                                modifier = Modifier.fillMaxWidth().height(8.dp)
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text("Synced: 248 photos", style = Typography.bodyMedium, color = Bone)
                                Text("Queued: 3 remaining", style = Typography.bodyMedium, color = Fresh)
                            }
                        }

                        // Start/Stop action buttons
                        Button(
                            onClick = {},
                            shape = RoundedCornerShape(percent = 100),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Fresh,
                                contentColor = Ink
                            ),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("RUN SYNC ENGINE", style = Typography.labelLarge)
                        }
                    }
                }
            }
        }
    }
}
