package com.quickpitik.mobile.ui.photographer

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Place
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.EarningsOverviewDto
import com.quickpitik.mobile.data.remote.PayoutBalanceDto
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.data.remote.PhotographerTransactionDto
import com.quickpitik.mobile.ui.theme.ArrowLabel
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.ErrorView
import com.quickpitik.mobile.ui.theme.FieldShape
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.LoadingSkeleton
import com.quickpitik.mobile.ui.theme.NumeralStyle
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.StatusChip
import com.quickpitik.mobile.ui.theme.StatusTone
import com.quickpitik.mobile.ui.theme.SuccessGreen
import com.quickpitik.mobile.ui.theme.TileShape
import com.quickpitik.mobile.ui.theme.Typography
import com.quickpitik.mobile.ui.theme.WarningOrange

private data class SetupStep(
    val title: String,
    val isCompleted: Boolean,
    val icon: ImageVector,
)

@Composable
fun PhotographerOverviewScreen(
    viewModel: PhotographerDashboardViewModel,
    onNavigateToSettings: () -> Unit,
    // Studio ROUTE name (e.g. "studio/capture") — the tabs are NavHost routes
    // now, and MainActivity's studioNavigate owns the per-tab refresh.
    onNavigateToTab: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val verificationState by viewModel.verificationState.collectAsState()
    val eventsState by viewModel.eventsState.collectAsState()
    val earningsUiState by viewModel.earningsUiState.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val context = LocalContext.current
    val sessionManager = remember { SessionManager.getInstance(context) }
    val photographerName = sessionManager.getUserName() ?: "Photographer"
    val firstName = photographerName.split("\\s+".toRegex()).firstOrNull() ?: "there"

    // Initial fetches owned by PhotographerDashboardScreen on first launch + the
    // Home nav-item onClick on tab re-tap. Overview re-fetching here is redundant
    // and contributes to a cold-start I/O storm (see 2026-05-27 home redesign).

    val latestMessage = remember(messages) { messages.maxByOrNull { it.createdAt } }
    val currentStatus = (verificationState as? VerificationUiState.Success)
        ?.verification?.status?.lowercase() ?: "incomplete"
    val isRejected = (currentStatus == "incomplete" || currentStatus == "rejected") &&
        (latestMessage?.kind == "verification_rejected")
    val rejectionReason = if (isRejected) latestMessage?.body else null
    val isApproved = currentStatus == "approved"

    val events = remember(eventsState) {
        when (val state = eventsState) {
            is EventsState.Success -> state.events
            else -> emptyList()
        }
    }
    val earnings = (earningsUiState as? EarningsUiState.Success)?.overview
    val transactions = (earningsUiState as? EarningsUiState.Success)?.transactions ?: emptyList()

    val verification = (verificationState as? VerificationUiState.Success)?.verification
    val missingList = verification?.missing ?: emptyList()
    val setupItems = buildSetupItems(missingList)
    val completedCount = setupItems.count { it.isCompleted }

    val chipTone = when {
        isApproved -> StatusTone.Approved
        currentStatus == "pending" -> StatusTone.Warning
        isRejected || currentStatus == "rejected" -> StatusTone.Danger
        else -> StatusTone.Warning
    }
    val chipText = when {
        isApproved -> "Studio live"
        currentStatus == "pending" -> "Review in progress"
        isRejected || currentStatus == "rejected" -> "Changes needed"
        else -> "Setup pending · $completedCount of ${setupItems.size}"
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Bone)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
    ) {
        if (isApproved) {
            StatusChip(text = chipText, tone = chipTone)
        } else {
            Surface(
                shape = PillShape,
                color = Ink,
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    val dotColor = when {
                        isRejected || currentStatus == "rejected" -> ErrorRed
                        completedCount == setupItems.size -> Fresh
                        else -> WarningOrange
                    }
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .clip(CircleShape)
                            .background(dotColor),
                    )
                    Text(
                        text = when {
                            currentStatus == "pending" || isRejected || currentStatus == "rejected" -> chipText.uppercase()
                            completedCount == setupItems.size -> "READY FOR REVIEW"
                            else -> "SETUP PENDING · $completedCount OF ${setupItems.size}"
                        },
                        style = Typography.labelSmall.copy(fontWeight = FontWeight.Bold, letterSpacing = 0.5.sp),
                        color = Bone,
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = if (isApproved) "Welcome back, $firstName." else "Welcome, $firstName.",
            color = Ink,
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            lineHeight = 34.sp,
        )

        if (!isApproved) {
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = when {
                    currentStatus == "pending" ->
                        "An admin is reviewing your studio setup. Check back shortly."
                    isRejected || currentStatus == "rejected" ->
                        rejectionReason ?: "Some details need updates before runners can find you."
                    else ->
                        "Finish your ${setupItems.size} studio requirements below to start covering events and uploading photos."
                },
                color = Slate,
                fontSize = 14.sp,
                lineHeight = 20.sp,
            )
        }

        if (isRejected || currentStatus == "rejected") {
            Spacer(modifier = Modifier.height(14.dp))
            Surface(
                shape = QpCardShape,
                color = ErrorRed.copy(alpha = 0.08f),
                border = BorderStroke(1.dp, ErrorRed.copy(alpha = 0.3f)),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Row(
                    modifier = Modifier.padding(14.dp),
                    verticalAlignment = Alignment.Top,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Box(
                        modifier = Modifier
                            .size(24.dp)
                            .clip(CircleShape)
                            .background(ErrorRed.copy(alpha = 0.15f)),
                        contentAlignment = Alignment.Center,
                    ) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = null,
                            tint = ErrorRed,
                            modifier = Modifier.size(14.dp),
                        )
                    }
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Changes needed on studio setup",
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp,
                            color = Ink,
                        )
                        Spacer(modifier = Modifier.height(2.dp))
                        Text(
                            text = rejectionReason ?: "Some details need updates before runners can find you. Open settings below to make corrections.",
                            style = Typography.bodySmall,
                            color = Slate,
                            lineHeight = 16.sp,
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        if (isApproved) {
            // The money cards must never render a FAILED (or still-loading)
            // earnings fetch as real ₱0 figures — an error shows as an error.
            when (earningsUiState) {
                is EarningsUiState.Error -> ErrorView(
                    message = (earningsUiState as EarningsUiState.Error).message,
                    title = "Couldn't load your studio stats",
                    onRetry = { viewModel.fetchEarningsAndTransactions() },
                    modifier = Modifier.fillMaxWidth(),
                )
                is EarningsUiState.Loading -> LoadingSkeleton(
                    modifier = Modifier.fillMaxWidth().height(220.dp),
                )
                is EarningsUiState.Success -> ApprovedHomeBody(
                    events = events,
                    earnings = earnings,
                    balance = (earningsUiState as? EarningsUiState.Success)?.balance,
                    transactions = transactions,
                    onCapture = { onNavigateToTab("studio/capture") },
                    onOpenEvent = { onNavigateToTab("studio/events") },
                    onOpenEarnings = { onNavigateToTab("studio/earnings") },
                )
            }
        } else {
            SetupHomeBody(
                verificationState = verificationState,
                items = setupItems,
                completedCount = completedCount,
                isRejected = isRejected || currentStatus == "rejected",
                onNavigateToSettings = onNavigateToSettings,
                onRefresh = {
                    viewModel.fetchVerificationStatus()
                    viewModel.fetchEvents()
                    viewModel.fetchEarningsAndTransactions()
                    viewModel.fetchMessages()
                    viewModel.fetchSettings()
                },
            )
        }
    }
}

private fun buildSetupItems(missingList: List<String>): List<SetupStep> {
    fun has(token: String) = !missingList.any { it.lowercase().contains(token) }
    return listOf(
        SetupStep("Avatar", has("profile") && has("avatar"), Icons.Default.AccountCircle),
        SetupStep("Cover banner", has("cover"), Icons.Default.Add),
        SetupStep("Brand & bio", has("brand"), Icons.Default.Edit),
        SetupStep("DSLR watermark", has("watermark"), Icons.Default.Star),
        SetupStep("GCash payout", has("payout"), Icons.Default.ShoppingCart),
        SetupStep("Public handle", has("handle"), Icons.Default.Face),
        SetupStep("Region setup", has("region"), Icons.Default.Place),
        SetupStep("Social media", has("social"), Icons.Default.Share),
    )
}

@Composable
private fun ApprovedHomeBody(
    events: List<PhotographerEventSummaryDto>,
    earnings: EarningsOverviewDto?,
    balance: PayoutBalanceDto?,
    transactions: List<PhotographerTransactionDto>,
    onCapture: () -> Unit,
    onOpenEvent: () -> Unit,
    onOpenEarnings: () -> Unit,
) {
    val coveredCount = events.size
    val thisWeek = earnings?.thisWeek ?: 0.0
    val sparkData = remember(earnings) { earnings?.weeklySeries?.map { it.amount } ?: emptyList() }
    val upcomingEvents = remember(events) {
        events.filter { it.state.lowercase() == "upcoming" }.sortedBy { it.date }
    }

    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        // Two stat cards — kept on Slate so Fresh stays on the CTA below.
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            QpCard(
                modifier = Modifier
                    .weight(1f)
                    .height(112.dp),
                padding = 14.dp,
            ) {
                Kicker(text = "Covered", color = SlateSoft)
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = coveredCount.toString(),
                    style = NumeralStyle.copy(fontSize = 32.sp),
                    color = Ink,
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = if (coveredCount == 1) "event" else "events",
                    color = SlateSoft,
                    fontSize = 12.sp,
                )
            }
            QpCard(
                modifier = Modifier
                    .weight(1f)
                    .height(112.dp),
                padding = 14.dp,
            ) {
                Kicker(text = "This week", color = SlateSoft)
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "₱${String.format("%,.0f", thisWeek)}",
                    style = NumeralStyle.copy(fontSize = 24.sp),
                    color = Ink,
                )
                if (sparkData.size >= 2) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Sparkline(
                        data = sparkData,
                        color = Slate,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(20.dp),
                    )
                }
            }
        }

        QpCard(padding = 16.dp) {
            Kicker(text = "Next up", color = SlateSoft)
            Spacer(modifier = Modifier.height(8.dp))
            if (upcomingEvents.isEmpty()) {
                Text(
                    text = "No upcoming coverage.",
                    color = Ink,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Once organizers assign you to a future event, it shows up here.",
                    color = SlateSoft,
                    fontSize = 12.sp,
                    lineHeight = 16.sp,
                )
            } else {
                // Every upcoming event, not just the soonest (web "Next up"
                // slab parity) — capped at 3 rows with a "+N more" line
                // instead of load-more machinery; the Events tab has the rest.
                upcomingEvents.take(3).forEachIndexed { index, event ->
                    if (index > 0) {
                        Spacer(modifier = Modifier.height(10.dp))
                        Divider(color = Line)
                        Spacer(modifier = Modifier.height(10.dp))
                    }
                    Text(
                        text = event.date,
                        color = SlateSoft,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = event.name,
                        color = Ink,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        lineHeight = 22.sp,
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = event.location,
                        color = SlateSoft,
                        fontSize = 12.sp,
                    )
                }
                if (upcomingEvents.size > 3) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Kicker(
                        text = "+${upcomingEvents.size - 3} more upcoming",
                        color = SlateSoft,
                    )
                }
                Spacer(modifier = Modifier.height(14.dp))
                GhostCta(
                    text = "Open events",
                    onClick = onOpenEvent,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        Spacer(modifier = Modifier.height(4.dp))

        // The single Fresh element in the viewport.
        PrimaryCta(
            text = "Capture photos",
            onClick = onCapture,
            modifier = Modifier.fillMaxWidth(),
        )

        Spacer(modifier = Modifier.height(8.dp))

        // 3-tile studio metrics row. Slate-only — Fresh stays on the CTA above.
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            MetricTile(
                kicker = "Sold",
                value = (earnings?.thisMonthSold ?: 0L).toString(),
                hint = "this month",
                modifier = Modifier.weight(1f),
            )
            MetricTile(
                kicker = "This month",
                value = "₱${String.format("%,.0f", earnings?.thisMonth ?: 0.0)}",
                hint = "earned",
                modifier = Modifier.weight(1f),
            )
            MetricTile(
                kicker = "Lifetime",
                value = "₱${String.format("%,.0f", earnings?.lifetimeKept ?: 0.0)}",
                hint = "kept",
                modifier = Modifier.weight(1f),
            )
        }

        // Billing glance — web dashboard parity: the payout request's state is
        // visible from Home, so a held request isn't discovered days later
        // deep in the Earnings tab. Three states: open request / ready to
        // request / balance building.
        if (balance != null) {
            QpCard(padding = 16.dp) {
                Kicker(text = "Billing", color = SlateSoft)
                Spacer(modifier = Modifier.height(8.dp))
                val open = balance.openRequest
                val (line, tone) = when {
                    open != null && open.status.uppercase() == "HELD" ->
                        "Payout held — action needed" to StatusTone.Danger
                    open != null ->
                        "Payout ${open.status.lowercase().replace('_', ' ')}" to StatusTone.Warning
                    balance.unpaidBalance >= balance.minimum ->
                        "Ready to request" to StatusTone.Approved
                    else ->
                        "Balance building" to StatusTone.Neutral
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    StatusChip(text = line, tone = tone)
                    Text(
                        text = "₱%,.2f".format(open?.amount ?: balance.unpaidBalance),
                        style = NumeralStyle.copy(fontSize = 16.sp, fontWeight = FontWeight.SemiBold),
                        color = Ink,
                    )
                }
                Spacer(modifier = Modifier.height(10.dp))
                ArrowLabel(
                    text = "Open earnings →",
                    color = Ink,
                    modifier = Modifier.clickable { onOpenEarnings() },
                )
            }
        }

        // Recent sales — last 3 transactions, empty state if none.
        QpCard(padding = 18.dp) {
            Kicker(text = "Recent sales", color = SlateSoft)
            Spacer(modifier = Modifier.height(12.dp))
            if (transactions.isEmpty()) {
                Text(
                    text = "No sales yet.",
                    color = Ink,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Your first sale shows up here once a runner buys one of your photos.",
                    color = SlateSoft,
                    fontSize = 12.sp,
                    lineHeight = 16.sp,
                )
            } else {
                transactions.take(3).forEachIndexed { index, tx ->
                    if (index > 0) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Divider(color = Line)
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.Top,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = tx.eventName ?: "Event",
                                color = Ink,
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                maxLines = 1,
                            )
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = formatPaidAt(tx.paidAt),
                                color = SlateSoft,
                                fontSize = 11.sp,
                            )
                        }
                        Text(
                            text = "₱${String.format("%,.0f", tx.amountKept)}",
                            color = Ink,
                            style = NumeralStyle.copy(fontSize = 16.sp),
                        )
                    }
                }
                Spacer(modifier = Modifier.height(14.dp))
                GhostCta(
                    text = "Open earnings",
                    onClick = onOpenEarnings,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}

@Composable
private fun MetricTile(
    kicker: String,
    value: String,
    hint: String,
    modifier: Modifier = Modifier,
) {
    QpCard(modifier = modifier.height(96.dp), padding = 12.dp) {
        Kicker(text = kicker, color = SlateSoft)
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = value,
            style = NumeralStyle.copy(fontSize = 20.sp),
            color = Ink,
            maxLines = 1,
        )
        Spacer(modifier = Modifier.height(2.dp))
        Text(
            text = hint,
            color = SlateSoft,
            fontSize = 10.sp,
        )
    }
}

private fun formatPaidAt(iso: String): String {
    return try {
        val parts = iso.substring(0, 10).split("-")
        val months = listOf(
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
            "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
        )
        val month = months[parts[1].toInt() - 1]
        "$month ${parts[2].toInt()}"
    } catch (e: Exception) {
        iso
    }
}

@Composable
private fun SetupHomeBody(
    verificationState: VerificationUiState,
    items: List<SetupStep>,
    completedCount: Int,
    isRejected: Boolean,
    onNavigateToSettings: () -> Unit,
    onRefresh: () -> Unit,
) {
    when (verificationState) {
        is VerificationUiState.Loading -> {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(220.dp),
                contentAlignment = Alignment.Center,
            ) {
                CircularProgressIndicator(color = Fresh, strokeWidth = 3.dp)
            }
        }
        is VerificationUiState.Error -> {
            ErrorView(
                message = verificationState.message,
                onRetry = onRefresh,
            )
        }
        is VerificationUiState.Success -> {
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                QpCard(padding = 20.dp) {
                    // Header Row: Kicker & Percent Pill
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column {
                            Kicker(text = "Studio Launch Checklist", color = SlateSoft)
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "$completedCount of ${items.size} completed",
                                color = Ink,
                                style = Typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                            )
                        }
                        Surface(
                            shape = PillShape,
                            color = if (completedCount == items.size) Fresh.copy(alpha = 0.12f) else Line.copy(alpha = 0.6f),
                        ) {
                            Text(
                                text = "${completedCount * 100 / items.size}%",
                                color = if (completedCount == items.size) Fresh else Ink,
                                style = Typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    // Custom animated progress bar — no M3 stop dot artifact!
                    val animatedProgress by animateFloatAsState(
                        targetValue = completedCount.toFloat() / items.size.toFloat(),
                        animationSpec = tween(400),
                        label = "setupProgress",
                    )
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .clip(PillShape)
                            .background(Line.copy(alpha = 0.5f)),
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth(fraction = animatedProgress.coerceIn(0f, 1f))
                                .fillMaxHeight()
                                .clip(PillShape)
                                .background(if (isRejected) ErrorRed else Fresh),
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    // Explanatory helper caption
                    val helperText = when {
                        completedCount == items.size -> "All requirements completed! Submit for admin verification in settings."
                        completedCount == 0 -> "Complete these requirements to start covering races and selling photos."
                        else -> "${items.size - completedCount} requirements left before your studio can be approved."
                    }
                    Text(
                        text = helperText,
                        color = Slate,
                        style = Typography.bodySmall,
                        lineHeight = 16.sp,
                    )

                    Spacer(modifier = Modifier.height(18.dp))

                    // 2-column grid of SetupTile
                    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        for (i in items.indices step 2) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                            ) {
                                SetupTile(
                                    step = items[i],
                                    onClick = onNavigateToSettings,
                                    modifier = Modifier.weight(1f),
                                )
                                if (i + 1 < items.size) {
                                    SetupTile(
                                        step = items[i + 1],
                                        onClick = onNavigateToSettings,
                                        modifier = Modifier.weight(1f),
                                    )
                                } else {
                                    Spacer(modifier = Modifier.weight(1f))
                                }
                            }
                        }
                    }
                }

                PrimaryCta(
                    text = "Open settings →",
                    onClick = onNavigateToSettings,
                    modifier = Modifier.fillMaxWidth(),
                )
                GhostCta(
                    text = "Sync status",
                    onClick = onRefresh,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}

@Composable
private fun SetupTile(
    step: SetupStep,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val isDone = step.isCompleted
    val borderColor = if (isDone) SuccessGreen.copy(alpha = 0.35f) else Line
    val backgroundColor = if (isDone) SuccessGreen.copy(alpha = 0.05f) else Bone
    val iconBg = if (isDone) SuccessGreen.copy(alpha = 0.15f) else BoneDeep
    val iconTint = if (isDone) SuccessGreen else Slate
    Row(
        modifier = modifier
            .heightIn(min = 64.dp)
            .clip(QpCardShape)
            .background(backgroundColor)
            .border(BorderStroke(1.dp, borderColor), QpCardShape)
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Box(
            modifier = Modifier
                .size(34.dp)
                .clip(CircleShape)
                .background(iconBg)
                .border(BorderStroke(1.dp, if (isDone) SuccessGreen.copy(alpha = 0.25f) else Line), CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                imageVector = step.icon,
                contentDescription = step.title,
                tint = iconTint,
                modifier = Modifier.size(16.dp),
            )
        }
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                text = step.title,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = Ink,
                maxLines = 1,
            )
            Spacer(modifier = Modifier.height(2.dp))
            if (isDone) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(3.dp),
                ) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = "Done",
                        tint = SuccessGreen,
                        modifier = Modifier.size(12.dp),
                    )
                    Text(
                        text = "Complete",
                        fontSize = 11.sp,
                        color = SuccessGreen,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
            } else {
                Text(
                    text = "Required",
                    fontSize = 11.sp,
                    color = SlateSoft,
                    fontWeight = FontWeight.Medium,
                )
            }
        }
    }
}

@Composable
private fun Sparkline(
    data: List<Double>,
    color: Color,
    modifier: Modifier = Modifier,
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
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(
            path = path,
            color = color,
            style = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round),
        )
    }
}
