package com.quickpitik.mobile.ui.photographer

import androidx.compose.animation.core.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.*
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
    onNavigateToTab: (Int) -> Unit,
    onPreviewProfile: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val verificationState by viewModel.verificationState.collectAsState()
    val eventsState by viewModel.eventsState.collectAsState()
    val earningsUiState by viewModel.earningsUiState.collectAsState()
    val brandSettings by viewModel.brandSettings.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val context = LocalContext.current
    val sessionManager = remember { SessionManager.getInstance(context) }
    val photographerName = sessionManager.getUserName() ?: "Photographer"

    // Notifications state
    var showNotifDialog by remember { mutableStateOf(false) }

    // Derive rejection from real inbox messages
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

    val events = remember(eventsState) {
        when (val state = eventsState) {
            is EventsState.Success -> state.events
            else -> emptyList()
        }
    }
    val earnings = (earningsUiState as? EarningsUiState.Success)?.overview
    val balance = (earningsUiState as? EarningsUiState.Success)?.balance

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Bone)
            .padding(20.dp)
            .verticalScroll(rememberScrollState())
    ) {
        if (currentStatus == "approved") {
            // --- DATA MODE (Dynamic Web Mirror) ---
            
            // Upper dynamic metadata kicker
            Row(
                modifier = Modifier.fillMaxWidth().padding(bottom = 6.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "OVERVIEW · CEBU",
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
                    text = "Welcome back, ${photographerName.split("\\s+".toRegex()).firstOrNull() ?: "there"}.",
                    color = Ink,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f)
                )

                // Interactive Bell Button
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

            // 2x2 Action Grid
            DashboardActionGrid(
                events = events,
                brandSettings = brandSettings,
                photographerName = photographerName,
                earnings = earnings,
                onNavigateToTab = onNavigateToTab,
                onPreviewProfile = onPreviewProfile
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Billing slab card
            BillingGlance(
                balance = balance,
                onNavigateToTab = { onNavigateToTab(3) }
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Next-Up slab card
            NextUpGlance(
                events = events,
                onNavigateToTab = { onNavigateToTab(2) }
            )

        } else {
            // --- SETUP MODE (Dynamic Verification Checklist) ---
            
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

            // Dynamic Status Notifications
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

        Spacer(modifier = Modifier.height(16.dp))
        OutlinedButton(
            onClick = onPreviewProfile,
            colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink),
            border = BorderStroke(1.dp, Ink),
            shape = RoundedCornerShape(10.dp),
            modifier = Modifier.fillMaxWidth().height(48.dp)
        ) {
            Text("PREVIEW PUBLIC PROFILE", fontWeight = FontWeight.Bold, fontSize = 12.sp)
        }
    }

    // --- Interactive Notifications Inbox Dialog ---
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
private fun Sparkline(
    data: List<Double>,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier) {
        if (data.isEmpty()) return@Canvas
        val maxVal = data.maxOrNull() ?: 1.0
        val minVal = data.minOrNull() ?: 0.0
        val range = if (maxVal == minVal) 1.0 else maxVal - minVal
        val path = Path()
        val width = size.width
        val height = size.height
        val dx = if (data.size > 1) width / (data.size - 1) else width
        
        data.forEachIndexed { index, value ->
            val x = index * dx
            val y = height - ((value - minVal) / range).toFloat() * height
            if (index == 0) {
                path.moveTo(x, y)
            } else {
                path.lineTo(x, y)
            }
        }
        drawPath(
            path = path,
            color = Fresh,
            style = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round)
        )
    }
}

@Composable
private fun Slab(
    number: String,
    title: String,
    trailing: @Composable () -> Unit = {},
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        shape = RoundedCornerShape(20.dp),
        border = BorderStroke(1.dp, Line),
        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = number,
                        color = Fresh,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp,
                        modifier = Modifier.padding(end = 8.dp)
                    )
                    Text(
                        text = title.uppercase(),
                        color = Ink,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 0.5.sp
                    )
                }
                trailing()
            }
            content()
        }
    }
}

@Composable
private fun DashboardActionGrid(
    events: List<PhotographerEventSummaryDto>,
    brandSettings: com.quickpitik.mobile.data.remote.BrandSettingsResponseDto?,
    photographerName: String,
    earnings: EarningsOverviewDto?,
    onNavigateToTab: (Int) -> Unit,
    onPreviewProfile: () -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            UploadCard(modifier = Modifier.weight(1f), onClick = { onNavigateToTab(1) })
            ProfileCard(
                brandSettings = brandSettings,
                photographerName = photographerName,
                modifier = Modifier.weight(1f),
                onClick = onPreviewProfile
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            EarningsCard(
                earnings = earnings,
                modifier = Modifier.weight(1f),
                onClick = { onNavigateToTab(3) }
            )
            EventsCard(
                events = events,
                modifier = Modifier.weight(1f),
                onClick = { onNavigateToTab(2) }
            )
        }
    }
}

@Composable
private fun UploadCard(
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.dp, Line),
        modifier = modifier
            .height(130.dp)
            .clickable { onClick() }
    ) {
        Column(
            modifier = Modifier.padding(14.dp).fillMaxSize(),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                val colors = listOf(Ink, InkSoft, Slate, SlateSoft, Ink)
                colors.forEach { col ->
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(32.dp)
                            .clip(RoundedCornerShape(6.dp))
                            .background(col)
                    )
                }
            }
            
            Column {
                Text(
                    text = "+ Upload photos",
                    color = Ink,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "any Cebu race",
                        color = Fresh,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Icon(
                        imageVector = Icons.Default.ArrowForward,
                        contentDescription = "Go",
                        tint = Fresh,
                        modifier = Modifier.size(10.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun ProfileCard(
    brandSettings: com.quickpitik.mobile.data.remote.BrandSettingsResponseDto?,
    photographerName: String,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    val brandColorHex = remember(brandSettings) {
        val raw = brandSettings?.brandColor?.uppercase() ?: ""
        if (raw.startsWith("#")) {
            try { Color(android.graphics.Color.parseColor(raw)) } catch (e: Exception) { Fresh }
        } else {
            Fresh
        }
    }
    
    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.dp, Line),
        modifier = modifier
            .height(130.dp)
            .clickable { onClick() }
    ) {
        Column(
            modifier = Modifier.padding(14.dp).fillMaxSize(),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Column {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(5.dp)
                        .clip(RoundedCornerShape(100.dp))
                        .background(brandColorHex)
                )
                Spacer(modifier = Modifier.height(10.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    val resolvedAvatarUrl = brandSettings?.avatarUrl?.replace("localhost", "10.0.2.2")
                    Box(
                        modifier = Modifier
                            .size(32.dp)
                            .clip(CircleShape)
                            .background(Bone)
                            .border(1.dp, Line, CircleShape),
                        contentAlignment = Alignment.Center
                    ) {
                        if (!resolvedAvatarUrl.isNullOrEmpty()) {
                            AsyncImage(
                                model = resolvedAvatarUrl,
                                contentDescription = "Avatar",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                              )
                        } else {
                            Icon(
                                imageVector = Icons.Default.Person,
                                contentDescription = "No Avatar",
                                tint = SlateSoft,
                                modifier = Modifier.size(16.dp)
                            )
                        }
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = brandSettings?.brandName?.ifBlank { photographerName } ?: photographerName,
                        color = Ink,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        maxLines = 1,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
            
            Column {
                Text(
                    text = "Open profile",
                    color = Ink,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
                val handleText = brandSettings?.handle?.ifBlank { "set URL" } ?: "set URL"
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = if (handleText == "set URL") handleText else "quickpitik.com/$handleText",
                        color = SlateSoft,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium,
                        maxLines = 1,
                        modifier = Modifier.weight(1f, fill = false)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Icon(
                        imageVector = Icons.Default.ArrowForward,
                        contentDescription = "Go",
                        tint = SlateSoft,
                        modifier = Modifier.size(10.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun EarningsCard(
    earnings: EarningsOverviewDto?,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.dp, Line),
        modifier = modifier
            .height(130.dp)
            .clickable { onClick() }
    ) {
        Column(
            modifier = Modifier.padding(14.dp).fillMaxSize(),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            val weeklyAmounts = remember(earnings) {
                earnings?.weeklySeries?.map { it.amount } ?: emptyList()
            }
            
            if (weeklyAmounts.size >= 2) {
                Sparkline(
                    data = weeklyAmounts,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(36.dp)
                )
            } else {
                Canvas(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(36.dp)
                ) {
                    val path = Path()
                    path.moveTo(0f, size.height * 0.7f)
                    path.lineTo(size.width * 0.3f, size.height * 0.5f)
                    path.lineTo(size.width * 0.6f, size.height * 0.8f)
                    path.lineTo(size.width, size.height * 0.2f)
                    drawPath(
                        path = path,
                        color = Line,
                        style = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round)
                    )
                }
            }
            
            Column {
                Text(
                    text = "View earnings",
                    color = Ink,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "₱${String.format("%,.0f", earnings?.thisWeek ?: 0.0)} this week",
                        color = SlateSoft,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Icon(
                        imageVector = Icons.Default.ArrowForward,
                        contentDescription = "Go",
                        tint = SlateSoft,
                        modifier = Modifier.size(10.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun EventsCard(
    events: List<PhotographerEventSummaryDto>,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    val live = remember(events) { events.filter { it.state.lowercase() == "live" }.size }
    val upcoming = remember(events) { events.filter { it.state.lowercase() == "upcoming" }.size }
    val archived = remember(events) { events.filter { it.state.lowercase() == "open" || it.state.lowercase() == "past" }.size }
    
    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.dp, Line),
        modifier = modifier
            .height(130.dp)
            .clickable { onClick() }
    ) {
        Column(
            modifier = Modifier.padding(14.dp).fillMaxSize(),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .clip(CircleShape)
                            .background(if (live > 0) Fresh else Line)
                    )
                    Text(
                        text = "$live Live",
                        color = Ink,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .clip(CircleShape)
                            .background(SlateSoft)
                    )
                    Text(
                        text = "$upcoming Upcoming",
                        color = Ink,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .clip(CircleShape)
                            .background(Line)
                    )
                    Text(
                        text = "$archived Archived",
                        color = Ink,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
            
            Column {
                Text(
                    text = "Manage events",
                    color = Ink,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "your covered races",
                        color = SlateSoft,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Icon(
                        imageVector = Icons.Default.ArrowForward,
                        contentDescription = "Go",
                        tint = SlateSoft,
                        modifier = Modifier.size(10.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun BillingGlance(
    balance: PayoutBalanceDto?,
    onNavigateToTab: () -> Unit
) {
    Slab(
        number = "01",
        title = "Billing",
        trailing = {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.clickable { onNavigateToTab() }
            ) {
                Text(
                    text = "Open billing",
                    color = SlateSoft,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.width(4.dp))
                Icon(
                    imageVector = Icons.Default.ArrowForward,
                    contentDescription = "Open billing",
                    tint = SlateSoft,
                    modifier = Modifier.size(12.dp)
                )
            }
        }
    ) {
        if (balance == null) {
            CircularProgressIndicator(color = Fresh, modifier = Modifier.size(24.dp))
        } else if (balance.unpaidBalance == 0.0 && !balance.hasOpenRequest) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .border(BorderStroke(1.dp, Line), RoundedCornerShape(12.dp))
                    .padding(20.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "No earnings yet.",
                        color = Ink,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Your first sale opens up the request flow — reach ₱${String.format("%,.0f", balance.minimum)} to request a payout.",
                        color = SlateSoft,
                        fontSize = 12.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }
        } else if (balance.hasOpenRequest && balance.openRequest != null) {
            val request = balance.openRequest
            val statusText = when (request.status.lowercase()) {
                "settled", "paid" -> "Approved · payment incoming"
                "held" -> "Held — needs attention"
                else -> "Pending review"
            }
            val statusColor = when (request.status.lowercase()) {
                "settled", "paid" -> Fresh
                "held" -> ErrorRed
                else -> Ink
            }
            
            Text(
                text = "Payout Request · $statusText",
                color = statusColor,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.5.sp
            )
            Text(
                text = "₱${String.format("%,.0f", request.amount)}",
                color = Ink,
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(top = 4.dp)
            )
            if (request.holdReason != null && request.status.lowercase() == "held") {
                Text(
                    text = "Reason: ${request.holdReason}",
                    color = ErrorRed,
                    fontSize = 12.sp,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
        } else {
            val eligible = balance.unpaidBalance >= balance.minimum
            Text(
                text = "Available to request",
                color = SlateSoft,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.5.sp
            )
            Text(
                text = "₱${String.format("%,.0f", balance.unpaidBalance)}",
                color = if (eligible) Fresh else SlateSoft,
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(top = 4.dp)
            )
            Text(
                text = if (eligible) "Ready to request — open billing to confirm and submit."
                       else "Reach ₱${String.format("%,.0f", balance.minimum)} in unpaid earnings to request a payout.",
                color = SlateSoft,
                fontSize = 12.sp,
                modifier = Modifier.padding(top = 8.dp)
            )
        }
    }
}

@Composable
private fun NextUpGlance(
    events: List<PhotographerEventSummaryDto>,
    onNavigateToTab: () -> Unit
) {
    val upcomingEvents = remember(events) {
        events.filter { it.state.lowercase() == "upcoming" }
            .sortedBy { it.date }
    }
    
    Slab(
        number = "02",
        title = "Next up",
        trailing = {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.clickable { onNavigateToTab() }
            ) {
                Text(
                    text = "All events",
                    color = SlateSoft,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.width(4.dp))
                Icon(
                    imageVector = Icons.Default.ArrowForward,
                    contentDescription = "All events",
                    tint = SlateSoft,
                    modifier = Modifier.size(12.dp)
                )
            }
        }
    ) {
        if (upcomingEvents.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .border(BorderStroke(1.dp, Line), RoundedCornerShape(12.dp))
                    .padding(20.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "No upcoming coverage.",
                        color = Ink,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Once organizers assign you to a future event, it shows up here.",
                        color = SlateSoft,
                        fontSize = 12.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }
        } else {
            Column(
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                upcomingEvents.take(3).forEach { event ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .border(BorderStroke(1.dp, Line), RoundedCornerShape(12.dp))
                            .background(BoneDeep.copy(alpha = 0.5f))
                            .padding(14.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = event.date,
                                color = SlateSoft,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = event.name,
                                color = Ink,
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                maxLines = 1,
                                modifier = Modifier.padding(top = 2.dp)
                            )
                        }
                        Button(
                            onClick = onNavigateToTab,
                            colors = ButtonDefaults.buttonColors(containerColor = Fresh),
                            shape = RoundedCornerShape(8.dp),
                            contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp),
                            modifier = Modifier.height(32.dp)
                        ) {
                            Text("Open", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 11.sp)
                        }
                    }
                }
            }
        }
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

    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        border = BorderStroke(1.dp, Line),
        shape = RoundedCornerShape(20.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            when (status) {
                "approved" -> {
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
                else -> {
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

            if (status != "approved") {
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
                val progressValue = if (isRejected) {
                    minOf(completedCount.toFloat() / items.size.toFloat(), 0.85f)
                } else {
                    completedCount.toFloat() / items.size.toFloat()
                }

                val progressColor = if (isRejected) ErrorRed else Fresh

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
