package com.quickpitik.mobile.ui.photographer

import androidx.compose.animation.core.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.VerificationSubmitResponseDto
import com.quickpitik.mobile.ui.theme.*

data class NotifItem(
    val id: String,
    val status: String,
    val title: String,
    val message: String,
    val timestamp: String,
    val isRead: Boolean
)

data class SetupStep(
    val title: String,
    val isCompleted: Boolean,
    val icon: ImageVector
)

@Composable
fun PhotographerOverviewScreen(
    viewModel: PhotographerDashboardViewModel,
    onNavigateToSettings: () -> Unit,
    modifier: Modifier = Modifier
) {
    val verificationState by viewModel.verificationState.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val context = LocalContext.current
    val sessionManager = remember { SessionManager.getInstance(context) }
    val photographerName = sessionManager.getUserName() ?: "Photographer"

    // Notifications state
    var showNotifDialog by remember { mutableStateOf(false) }

    // Derive rejection from real inbox messages:
    // If the latest message is a rejection and status is incomplete/rejected, treat onboarding as rejected.
    val latestMessage = remember(messages) { messages.maxByOrNull { it.createdAt } }
    val currentStatus = (verificationState as? VerificationUiState.Success)?.verification?.status?.lowercase() ?: "incomplete"
    val isRejected = (currentStatus == "incomplete" || currentStatus == "rejected") && (latestMessage?.kind == "verification_rejected")
    val rejectionReason = if (isRejected) latestMessage?.body else null

    // Build notificationList from real backend messages
    val notificationList = remember(messages) {
        messages.map { msg ->
            val status = if (msg.kind == "verification_rejected") "rejected" else if (msg.kind == "verification_approved") "approved" else "info"
            NotifItem(
                id = msg.id,
                status = status,
                title = if (msg.kind == "verification_rejected") "Profile Changes Required" else if (msg.kind == "verification_approved") "Onboarding Profile Approved!" else msg.title ?: "Notification",
                message = msg.body,
                timestamp = "System Message",
                isRead = msg.readAt != null
            )
        }
    }

    val unreadCount = remember(messages) {
        messages.count { it.readAt == null }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Bone)
            .padding(20.dp)
            .verticalScroll(rememberScrollState())
    ) {
        // Upper dynamic metadata kicker
        Row(
            modifier = Modifier.fillMaxWidth().padding(bottom = 6.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = "CEBU SINCE 2026",
                color = SlateSoft,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.sp
            )
            Text(
                text = "ONLINE SESSIONS ACTIVE",
                color = Fresh,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.sp
            )
        }

        // Dashboard Header Title & Notification Bell Icon Row
        Row(
            modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Welcome, $photographerName",
                color = Ink,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )

            // Interactive Bell Button with Red Unread Counter Badge
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(CircleShape)
                    .background(BoneDeep)
                    .border(1.dp, Line, CircleShape)
                    .clickable {
                        showNotifDialog = true
                        viewModel.markAllMessagesAsRead()
                    },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Notifications,
                    contentDescription = "Notifications Inbox",
                    tint = if (unreadCount > 0) Fresh else SlateSoft,
                    modifier = Modifier.size(22.dp)
                )

                if (unreadCount > 0) {
                    Box(
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(top = 4.dp, end = 4.dp)
                            .size(18.dp)
                            .clip(CircleShape)
                            .background(ErrorRed),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = unreadCount.toString(),
                            color = Color.White,
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }

        // Dynamic Status Notifications (Alerts at the top)
        if (verificationState is VerificationUiState.Success) {
            val status = if (isRejected) "rejected" else (verificationState as VerificationUiState.Success).verification.status.lowercase()
            val finalRejectionReason = if (isRejected) rejectionReason else (verificationState as VerificationUiState.Success).verification.suspensionReason
            if (status == "approved") {
                Card(
                    colors = CardDefaults.cardColors(containerColor = Fresh.copy(alpha = 0.1f)),
                    border = BorderStroke(1.dp, Fresh.copy(alpha = 0.4f)),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.CheckCircle,
                            contentDescription = "Approved",
                            tint = Fresh,
                            modifier = Modifier.size(24.dp)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Column {
                            Text(
                                text = "ONBOARDING APPROVED",
                                color = Fresh,
                                fontWeight = FontWeight.Bold,
                                fontSize = 11.sp,
                                letterSpacing = 0.5.sp
                            )
                            Text(
                                text = "Your professional studio is verified and live! You can now start syncing DSLR photos.",
                                color = Ink,
                                fontSize = 12.sp
                            )
                        }
                    }
                }
            } else if (status == "rejected") {
                Card(
                    colors = CardDefaults.cardColors(containerColor = ErrorRed.copy(alpha = 0.08f)),
                    border = BorderStroke(1.dp, ErrorRed.copy(alpha = 0.3f)),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = "Rejected",
                            tint = ErrorRed,
                            modifier = Modifier.size(24.dp)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "ONBOARDING REJECTED",
                                color = ErrorRed,
                                fontWeight = FontWeight.Bold,
                                fontSize = 11.sp,
                                letterSpacing = 0.5.sp
                            )
                            Text(
                                text = "Updates Required: " + (finalRejectionReason ?: "Review missing details."),
                                color = Ink,
                                fontSize = 12.sp
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Button(
                                onClick = onNavigateToSettings,
                                colors = ButtonDefaults.buttonColors(containerColor = ErrorRed),
                                shape = RoundedCornerShape(8.dp),
                                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp),
                                modifier = Modifier.height(32.dp)
                            ) {
                                Text("FIX ON SETTINGS PAGE", fontSize = 11.sp, color = Color.White, fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                }
            }
        }

        // Verification Panel State
        when (val state = verificationState) {
            is VerificationUiState.Loading -> {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = Fresh, strokeWidth = 3.dp)
                }
            }
            is VerificationUiState.Error -> {
                ErrorStateCard(message = state.message, onRetry = { viewModel.fetchVerificationStatus() })
            }
            is VerificationUiState.Success -> {
                val verification = state.verification
                VerificationPanel(
                    verification = verification,
                    isRejected = isRejected,
                    rejectionReason = rejectionReason,
                    onRefresh = { 
                        viewModel.fetchVerificationStatus()
                        viewModel.fetchMessages()
                    },
                    onNavigateToSettings = onNavigateToSettings
                )
            }
        }
    }

    // --- Interactive Notifications Dialog ---
    if (showNotifDialog) {
        AlertDialog(
            onDismissRequest = { showNotifDialog = false },
            confirmButton = {
                Button(
                    onClick = { showNotifDialog = false },
                    colors = ButtonDefaults.buttonColors(containerColor = Fresh),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text("Close Inbox", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                }
            },
            title = {
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Notifications,
                        contentDescription = "Notifications",
                        tint = Fresh,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Studio Notifications",
                        color = Ink,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            },
            text = {
                if (notificationList.isEmpty()) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 24.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(
                                imageVector = Icons.Default.Info,
                                contentDescription = "Empty",
                                tint = SlateSoft,
                                modifier = Modifier.size(36.dp)
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Your notifications inbox is empty.",
                                color = SlateSoft,
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                } else {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        notificationList.forEach { notif ->
                            val borderCol = if (notif.status == "approved") Fresh.copy(alpha = 0.3f) else ErrorRed.copy(alpha = 0.3f)
                            val bgCol = if (notif.status == "approved") Fresh.copy(alpha = 0.05f) else ErrorRed.copy(alpha = 0.05f)
                            
                            Card(
                                colors = CardDefaults.cardColors(containerColor = bgCol),
                                border = BorderStroke(1.dp, borderCol),
                                shape = RoundedCornerShape(10.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Row(
                                    modifier = Modifier.padding(12.dp),
                                    verticalAlignment = Alignment.Top
                                ) {
                                    Icon(
                                        imageVector = if (notif.status == "approved") Icons.Default.CheckCircle else Icons.Default.Warning,
                                        contentDescription = notif.status,
                                        tint = if (notif.status == "approved") Fresh else ErrorRed,
                                        modifier = Modifier.size(18.dp).padding(top = 2.dp)
                                    )
                                    Spacer(modifier = Modifier.width(10.dp))
                                    Column {
                                        Text(
                                            text = notif.title,
                                            color = Ink,
                                            fontWeight = FontWeight.Bold,
                                            fontSize = 13.sp
                                        )
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Text(
                                            text = notif.message,
                                            color = SlateSoft,
                                            fontSize = 12.sp,
                                            lineHeight = 16.sp
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            },
            containerColor = Bone,
            shape = RoundedCornerShape(16.dp)
        )
    }
}

@Composable
private fun ErrorStateCard(message: String, onRetry: () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        border = BorderStroke(1.dp, Line),
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp)
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = Icons.Default.Warning,
                contentDescription = "Error",
                tint = ErrorRed,
                modifier = Modifier.size(42.dp)
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = message,
                color = Ink,
                textAlign = TextAlign.Center,
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium
            )
            Spacer(modifier = Modifier.height(16.dp))
            Button(
                onClick = onRetry,
                colors = ButtonDefaults.buttonColors(containerColor = Fresh),
                shape = RoundedCornerShape(8.dp)
            ) {
                Text("Retry Sync Status", color = Color.White, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun VerificationPanel(
    verification: VerificationSubmitResponseDto,
    isRejected: Boolean,
    rejectionReason: String?,
    onRefresh: () -> Unit,
    onNavigateToSettings: () -> Unit
) {
    val status = if (isRejected) "rejected" else verification.status.lowercase()
    val missingList = verification.missing ?: emptyList()

    // 1. Account Suspension Banner
    if (verification.suspendedAt != null) {
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF1F2)),
            border = BorderStroke(1.dp, ErrorRed.copy(alpha = 0.4f)),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp)
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.Warning,
                    contentDescription = "Suspended",
                    tint = ErrorRed,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(12.dp))
                Column {
                    Text(
                        text = "ACCOUNT SUSPENDED",
                        color = ErrorRed,
                        fontWeight = FontWeight.Bold,
                        fontSize = 11.sp
                    )
                    Text(
                        text = verification.suspensionReason ?: "Your profile is restricted. Contact admin support.",
                        color = Ink,
                        fontSize = 12.sp
                    )
                }
            }
        }
    }

    // 2. Main Verification Onboarding Layout
    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        border = BorderStroke(1.dp, Line),
        shape = RoundedCornerShape(20.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            when (status) {
                "approved" -> {
                    // Beautiful Approved Badge
                    Row(
                        modifier = Modifier
                            .background(Fresh.copy(alpha = 0.1f), RoundedCornerShape(6.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.CheckCircle,
                            contentDescription = "Verified",
                            tint = SuccessGreen,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "VERIFIED PROFESSIONAL",
                            color = SuccessGreen,
                            fontWeight = FontWeight.Bold,
                            fontSize = 10.sp,
                            letterSpacing = 1.sp
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Your Studio is Live.",
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        color = Ink
                    )
                    Text(
                        text = "Runners can search your brand and buy your event photos! Capture and sync tethered files via the Tether tab.",
                        fontSize = 13.sp,
                        color = SlateSoft,
                        modifier = Modifier.padding(top = 4.dp, bottom = 20.dp)
                    )
                }
                "pending" -> {
                    // Review Pulsing Indicator
                    val infiniteTransition = rememberInfiniteTransition()
                    val pulseAlpha by infiniteTransition.animateFloat(
                        initialValue = 0.4f,
                        targetValue = 1.0f,
                        animationSpec = infiniteRepeatable(
                            animation = tween(1200, easing = LinearEasing),
                            repeatMode = RepeatMode.Reverse
                        )
                    )

                    Row(
                        modifier = Modifier
                            .background(Line, RoundedCornerShape(6.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(WarningOrange.copy(alpha = pulseAlpha))
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "REVIEW IN PROGRESS",
                            color = WarningOrange,
                            fontWeight = FontWeight.Bold,
                            fontSize = 10.sp,
                            letterSpacing = 1.sp
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Profile Verification Pending",
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        color = Ink
                    )
                    Text(
                        text = "An administrator is reviewing your brand metadata, payouts, and DSLR watermarks. Check back shortly.",
                        fontSize = 13.sp,
                        color = SlateSoft,
                        modifier = Modifier.padding(top = 4.dp, bottom = 20.dp)
                    )
                }
                "rejected" -> {
                    Row(
                        modifier = Modifier
                            .background(ErrorRed.copy(alpha = 0.1f), RoundedCornerShape(6.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = "Rejected",
                            tint = ErrorRed,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "CHANGES REQUIRED",
                            color = ErrorRed,
                            fontWeight = FontWeight.Bold,
                            fontSize = 10.sp,
                            letterSpacing = 1.sp
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Updates Needed",
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        color = Ink
                    )
                    Text(
                        text = rejectionReason ?: "Some details did not pass verification. Please correct your brand items in the Settings tab and submit again.",
                        fontSize = 13.sp,
                        color = SlateSoft,
                        modifier = Modifier.padding(top = 4.dp, bottom = 20.dp)
                    )
                }
                else -> { // "incomplete"
                    Text(
                        text = "ONBOARDING CHECKLIST",
                        color = SlateSoft,
                        fontWeight = FontWeight.Bold,
                        fontSize = 10.sp,
                        letterSpacing = 1.sp,
                        modifier = Modifier.padding(bottom = 6.dp)
                    )
                    Text(
                        text = "Complete Studio Setup",
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        color = Ink
                    )
                    Text(
                        text = "Complete the 8 requirements below. Approved setups go live immediately so runners can buy your captures.",
                        fontSize = 13.sp,
                        color = SlateSoft,
                        modifier = Modifier.padding(top = 4.dp, bottom = 16.dp)
                    )
                }
            }

            // Checklist & Progress Widget (Visible if incomplete or pending or rejected)
            if (status != "approved") {
                // 1. Dynamic Progress bar
                val hasAvatar = !missingList.any { it.lowercase().contains("profile") || it.lowercase().contains("avatar") }
                val hasCover = !missingList.any { it.lowercase().contains("cover") }
                val hasBrand = !missingList.any { it.lowercase().contains("brand") }
                val hasWatermark = !missingList.any { it.lowercase().contains("watermark") }
                val hasPayout = !missingList.any { it.lowercase().contains("payout") }
                val hasHandle = !missingList.any { it.lowercase().contains("handle") }
                val hasRegion = !missingList.any { it.lowercase().contains("region") }
                val hasSocial = !missingList.any { it.lowercase().contains("social") }

                val items = listOf(
                    SetupStep("Avatar", hasAvatar, Icons.Default.AccountCircle),
                    SetupStep("Cover Banner", hasCover, Icons.Default.Add),
                    SetupStep("Brand & Bio", hasBrand, Icons.Default.Edit),
                    SetupStep("DSLR Watermark", hasWatermark, Icons.Default.Star),
                    SetupStep("GCash Payout", hasPayout, Icons.Default.ShoppingCart),
                    SetupStep("Public Handle", hasHandle, Icons.Default.Face),
                    SetupStep("Region Setup", hasRegion, Icons.Default.Place),
                    SetupStep("Social Media", hasSocial, Icons.Default.Share)
                )

                val completedCount = items.count { it.isCompleted }
                
                // If rejected, cap the progress value so that it never shows 100% (indicating correction/fixes are needed)
                val progressValue = if (isRejected) {
                    minOf(completedCount.toFloat() / items.size.toFloat(), 0.85f)
                } else {
                    completedCount.toFloat() / items.size.toFloat()
                }

                val progressColor = if (isRejected) ErrorRed else Fresh

                // Progress Indicator Row
                Row(
                    modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = if (isRejected) "Review Corrections Needed" else "Setup Completed: $completedCount of ${items.size}",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = if (isRejected) ErrorRed else Ink
                    )
                    Text(
                        text = if (isRejected) "Needs Action" else "${(progressValue * 100).toInt()}%",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = progressColor
                    )
                }

                LinearProgressIndicator(
                    progress = progressValue,
                    color = progressColor,
                    trackColor = Line,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp))
                )

                Spacer(modifier = Modifier.height(20.dp))

                // 2. Interactive Setup Grid (2 Columns)
                Column {
                    for (i in items.indices step 2) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            GridItem(
                                step = items[i],
                                onClick = onNavigateToSettings,
                                modifier = Modifier.weight(1f)
                            )
                            if (i + 1 < items.size) {
                                GridItem(
                                    step = items[i + 1],
                                    onClick = onNavigateToSettings,
                                    modifier = Modifier.weight(1f)
                                )
                            } else {
                                Spacer(modifier = Modifier.weight(1f))
                            }
                        }
                        Spacer(modifier = Modifier.height(10.dp))
                    }
                }
                
                Spacer(modifier = Modifier.height(10.dp))
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Open Dashboard Settings Button
            OutlinedButton(
                onClick = onNavigateToSettings,
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Fresh),
                border = BorderStroke(1.dp, Fresh),
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth().height(48.dp)
            ) {
                Text(
                    text = "OPEN DASHBOARD SETTINGS",
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Refresh Status Action Button
            Button(
                onClick = onRefresh,
                colors = ButtonDefaults.buttonColors(containerColor = Fresh),
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth().height(48.dp)
            ) {
                Text(
                    text = "SYNC ONBOARDING STATUS",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp
                )
            }
        }
    }
}

@Composable
private fun GridItem(
    step: SetupStep,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = if (step.isCompleted) Fresh.copy(alpha = 0.06f) else Bone
        ),
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(
            1.dp, 
            if (step.isCompleted) Fresh.copy(alpha = 0.4f) else Line
        ),
        modifier = modifier
            .height(68.dp)
            .clickable { onClick() }
    ) {
        Row(
            modifier = Modifier.fillMaxSize().padding(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(28.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(if (step.isCompleted) Fresh.copy(alpha = 0.15f) else Line),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = step.icon,
                    contentDescription = step.title,
                    tint = if (step.isCompleted) Fresh else SlateSoft,
                    modifier = Modifier.size(16.dp)
                )
            }
            
            Spacer(modifier = Modifier.width(10.dp))
            
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = step.title,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    color = Ink
                )
                Text(
                    text = if (step.isCompleted) "Complete" else "Required",
                    fontSize = 10.sp,
                    color = if (step.isCompleted) Fresh else SlateSoft
                )
            }

            if (step.isCompleted) {
                Icon(
                    imageVector = Icons.Default.Check,
                    contentDescription = "Done",
                    tint = Fresh,
                    modifier = Modifier.size(14.dp)
                )
            }
        }
    }
}
