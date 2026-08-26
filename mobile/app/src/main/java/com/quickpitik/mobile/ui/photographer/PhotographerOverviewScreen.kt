package com.quickpitik.mobile.ui.photographer

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.*
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
import com.quickpitik.mobile.data.remote.*
import com.quickpitik.mobile.ui.theme.*

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
        StatusChip(text = chipText, tone = chipTone)

        Spacer(modifier = Modifier.height(20.dp))

        Text(
            text = if (isApproved) "Welcome back, $firstName." else "Welcome, $firstName.",
            color = Ink,
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            lineHeight = 34.sp,
        )

        if (!isApproved) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = when {
                    currentStatus == "pending" ->
                        "An admin is reviewing your studio setup. Check back shortly."
                    isRejected || currentStatus == "rejected" ->
                        rejectionReason ?: "Some details need updates before runners can find you."
                    else ->
                        "Finish your studio setup to start covering events."
                },
                color = Slate,
                fontSize = 14.sp,
                lineHeight = 20.sp,
            )
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
                QpCard(padding = 18.dp) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 10.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = "$completedCount of ${items.size} done",
                            color = Ink,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = "${completedCount * 100 / items.size}%",
                            color = Slate,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                        )
                    }
                    LinearProgressIndicator(
                        progress = completedCount.toFloat() / items.size.toFloat(),
                        color = if (isRejected) ErrorRed else Fresh,
                        trackColor = Line,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(6.dp)
                            .clip(TileShape),
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        for (i in items.indices step 2) {
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
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
                    text = "Open settings",
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
    val borderColor = if (step.isCompleted) SuccessGreen.copy(alpha = 0.4f) else Line
    val backgroundColor = if (step.isCompleted) SuccessGreen.copy(alpha = 0.06f) else Bone
    val tint = if (step.isCompleted) SuccessGreen else SlateSoft
    Row(
        modifier = modifier
            .height(60.dp)
            .background(backgroundColor, FieldShape)
            .border(BorderStroke(1.dp, borderColor), FieldShape)
            .clickable { onClick() }
            .padding(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .size(28.dp)
                .clip(TileShape)
                .background(if (step.isCompleted) SuccessGreen.copy(alpha = 0.15f) else Line),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                imageVector = step.icon,
                contentDescription = step.title,
                tint = tint,
                modifier = Modifier.size(16.dp),
            )
        }
        Spacer(modifier = Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.Center) {
            Text(
                text = step.title,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = Ink,
            )
            Text(
                text = if (step.isCompleted) "Complete" else "Required",
                fontSize = 10.sp,
                color = tint,
            )
        }
        if (step.isCompleted) {
            Icon(
                imageVector = Icons.Default.Check,
                contentDescription = "Done",
                tint = SuccessGreen,
                modifier = Modifier.size(14.dp),
            )
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
