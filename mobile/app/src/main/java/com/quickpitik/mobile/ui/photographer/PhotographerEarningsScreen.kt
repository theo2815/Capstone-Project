package com.quickpitik.mobile.ui.photographer

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.data.remote.EarningsOverviewDto
import com.quickpitik.mobile.data.remote.PayoutBalanceDto
import com.quickpitik.mobile.data.remote.PhotographerPayoutDto
import com.quickpitik.mobile.data.remote.PhotographerTransactionDto
import com.quickpitik.mobile.data.remote.WeeklyRevenuePointDto
import com.quickpitik.mobile.ui.theme.ArrowLabel
import com.quickpitik.mobile.ui.theme.BadgeShape
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.ErrorView
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.LoadingSkeleton
import com.quickpitik.mobile.ui.theme.NumeralStyle
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.SparklineBarShape
import com.quickpitik.mobile.ui.theme.StatusChip
import com.quickpitik.mobile.ui.theme.StatusTone
import com.quickpitik.mobile.ui.theme.Typography
import com.quickpitik.mobile.ui.theme.WarningOrange
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun PhotographerEarningsScreen(
    viewModel: PhotographerDashboardViewModel,
    modifier: Modifier = Modifier,
) {
    val earningsState by viewModel.earningsUiState.collectAsState()
    val payoutActionState by viewModel.payoutActionState.collectAsState()

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Bone)
            .padding(horizontal = 16.dp)
            .padding(top = 12.dp),
    ) {
        Kicker(text = "Earnings · kept after platform cut", color = Slate)
        Spacer(modifier = Modifier.height(20.dp))

        when (val state = earningsState) {
            is EarningsUiState.Loading -> EarningsLoadingSkeleton()
            is EarningsUiState.Error -> ErrorView(
                message = state.message,
                onRetry = { viewModel.fetchEarningsAndTransactions() },
            )
            is EarningsUiState.Success -> EarningsContent(
                overview = state.overview,
                balance = state.balance,
                transactions = state.transactions,
                payoutActionState = payoutActionState,
                onRequestPayout = { viewModel.submitPayoutRequest() },
                onClearActionState = { viewModel.clearPayoutActionState() },
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun EarningsContent(
    overview: EarningsOverviewDto,
    balance: PayoutBalanceDto,
    transactions: List<PhotographerTransactionDto>,
    payoutActionState: String?,
    onRequestPayout: () -> Unit,
    onClearActionState: () -> Unit,
) {
    var showHistory by remember { mutableStateOf(false) }
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(24.dp),
        contentPadding = PaddingValues(bottom = 32.dp),
    ) {
        item("lifetime") {
            LifetimeSlab(
                lifetimeKept = overview.lifetimeKept,
                weeklySeries = overview.weeklySeries,
            )
        }
        item("breakdown") {
            Divider(thickness = 1.dp, color = Line)
            Spacer(modifier = Modifier.height(24.dp))
            BreakdownSlab(overview = overview)
        }
        item("wallet") {
            Spacer(modifier = Modifier.height(24.dp))
            Divider(thickness = 1.dp, color = Line)
            Spacer(modifier = Modifier.height(24.dp))
            WalletSlab(
                balance = balance,
                payoutActionState = payoutActionState,
                onRequestPayout = onRequestPayout,
                onClearActionState = onClearActionState,
            )
        }
        item("transactionsLink") {
            Spacer(modifier = Modifier.height(24.dp))
            Divider(thickness = 1.dp, color = Line)
            Spacer(modifier = Modifier.height(8.dp))
            TransactionsLink(
                count = transactions.size,
                onClick = { showHistory = true },
            )
        }
    }

    if (showHistory) {
        ModalBottomSheet(
            onDismissRequest = { showHistory = false },
            sheetState = sheetState,
            containerColor = Bone,
        ) {
            TransactionsSheet(transactions = transactions)
        }
    }
}

@Composable
private fun LifetimeSlab(
    lifetimeKept: Double,
    weeklySeries: List<WeeklyRevenuePointDto>,
) {
    Column {
        Kicker(text = "Lifetime kept", color = SlateSoft)
        Spacer(modifier = Modifier.height(10.dp))
        Text(
            text = "₱%,.0f".format(lifetimeKept),
            style = NumeralStyle.copy(fontSize = 44.sp, fontWeight = FontWeight.SemiBold),
            color = Fresh,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "After platform cut",
            color = Slate,
            style = Typography.bodyMedium,
        )

        Spacer(modifier = Modifier.height(28.dp))
        Kicker(text = "12-week trend", color = SlateSoft)
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = "Tap a bar for week details",
            color = SlateSoft,
            style = Typography.bodySmall,
        )
        Spacer(modifier = Modifier.height(16.dp))

        if (weeklySeries.isEmpty()) {
            Text(
                text = "No earnings yet — your first weeks will plot here.",
                color = SlateSoft,
                style = Typography.bodySmall,
            )
        } else {
            Sparkline(data = weeklySeries)
        }
    }
}

@Composable
private fun Sparkline(
    data: List<WeeklyRevenuePointDto>,
    modifier: Modifier = Modifier,
) {
    val max = remember(data) { data.maxOf { it.amount }.coerceAtLeast(1.0) }
    var selectedIndex by remember(data) { mutableStateOf(data.lastIndex) }
    val selected = data.getOrNull(selectedIndex)

    Column(modifier = modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(64.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.Bottom,
        ) {
            data.forEachIndexed { index, point ->
                val targetHeight = (point.amount / max).toFloat().coerceAtLeast(0.04f)
                val animatedHeight by animateFloatAsState(
                    targetValue = targetHeight,
                    animationSpec = tween(durationMillis = 220),
                    label = "qp-sparkline-bar-h",
                )
                val animatedColor by animateColorAsState(
                    targetValue = if (index == selectedIndex) Ink else Line,
                    animationSpec = tween(durationMillis = 180),
                    label = "qp-sparkline-bar-c",
                )
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                        .clickable { selectedIndex = index },
                    contentAlignment = Alignment.BottomCenter,
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .fillMaxHeight(animatedHeight)
                            .clip(SparklineBarShape)
                            .background(animatedColor),
                    )
                }
            }
        }
        if (selected != null) {
            Spacer(modifier = Modifier.height(10.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Week of ${formatWeekLabel(selected.weekOf)}",
                    color = Slate,
                    style = Typography.bodySmall,
                )
                Text(
                    text = "₱%,.0f".format(selected.amount),
                    style = NumeralStyle.copy(fontSize = 14.sp, fontWeight = FontWeight.SemiBold),
                    color = Ink,
                )
            }
        }
    }
}

@Composable
private fun BreakdownSlab(overview: EarningsOverviewDto) {
    Column {
        Kicker(text = "Breakdown · current cycle", color = SlateSoft)
        Spacer(modifier = Modifier.height(16.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.Top,
        ) {
            BreakdownStat(
                kicker = "This week",
                value = "₱%,.0f".format(overview.thisWeek),
                caption = "+${overview.thisWeekSold} sold",
                modifier = Modifier.weight(1f),
            )
            Divider(
                color = Line,
                modifier = Modifier
                    .height(60.dp)
                    .width(1.dp),
            )
            BreakdownStat(
                kicker = "This month",
                value = "₱%,.0f".format(overview.thisMonth),
                caption = "+${overview.thisMonthSold} sold",
                modifier = Modifier.weight(1f),
            )
        }
        Spacer(modifier = Modifier.height(20.dp))
        Divider(thickness = 1.dp, color = Line)
        Spacer(modifier = Modifier.height(16.dp))
        Kicker(text = "Payout pending", color = SlateSoft)
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "₱%,.0f".format(overview.payoutPending),
            style = NumeralStyle.copy(fontSize = 22.sp, fontWeight = FontWeight.SemiBold),
            color = Ink,
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = "Releases ${overview.payoutScheduledFor ?: "on weekly review"}",
            color = Slate,
            style = Typography.bodySmall,
        )
    }
}

@Composable
private fun BreakdownStat(
    kicker: String,
    value: String,
    caption: String,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier) {
        Kicker(text = kicker, color = Slate)
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = value,
            style = NumeralStyle.copy(fontSize = 22.sp, fontWeight = FontWeight.SemiBold),
            color = Ink,
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = caption,
            color = Slate,
            style = Typography.bodySmall,
        )
    }
}

@Composable
private fun WalletSlab(
    balance: PayoutBalanceDto,
    payoutActionState: String?,
    onRequestPayout: () -> Unit,
    onClearActionState: () -> Unit,
) {
    // The kicker + CTA name the ACTUAL primary payout method from the
    // backend-authoritative balance snapshot — this card used to hardcode
    // "GCash" whatever the account was, and offered the CTA even when no
    // primary account existed (the demo-day PAYOUT_NO_ACCOUNT mismatch the
    // backend added `primaryAccount` to prevent).
    val methodLabel = when (balance.primaryAccount?.method?.lowercase()) {
        "gcash" -> "GCash"
        "maya" -> "Maya"
        "gotyme" -> "GoTyme"
        null -> null
        else -> balance.primaryAccount.method.replaceFirstChar { it.uppercase() }
    }
    QpCard(modifier = Modifier.fillMaxWidth(), padding = 16.dp) {
        Kicker(
            text = if (methodLabel != null) "Unpaid balance · $methodLabel" else "Unpaid balance",
            color = SlateSoft,
        )
        Spacer(modifier = Modifier.height(10.dp))
        Text(
            text = "₱%,.2f".format(balance.unpaidBalance),
            style = NumeralStyle.copy(fontSize = 32.sp, fontWeight = FontWeight.SemiBold),
            color = Ink,
        )
        Spacer(modifier = Modifier.height(16.dp))

        val hasOpen = balance.hasOpenRequest || balance.openRequest != null
        if (hasOpen) {
            OpenRequestBlock(openRequest = balance.openRequest, fallbackAmount = balance.unpaidBalance)
        } else {
            val hasAccount = balance.primaryAccount != null
            val canWithdraw = hasAccount && balance.unpaidBalance >= balance.minimum
            val isProcessing = payoutActionState == "processing"
            GhostCta(
                text = when {
                    isProcessing -> "Requesting…"
                    methodLabel != null -> "Request payout to $methodLabel"
                    else -> "Request payout"
                },
                onClick = onRequestPayout,
                enabled = canWithdraw && !isProcessing,
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = when {
                    !hasAccount ->
                        "Set up a payout account in Settings to request payouts."
                    canWithdraw ->
                        "Minimum withdrawal ₱%,.0f".format(balance.minimum)
                    else ->
                        "Reach ₱%,.0f to request a payout".format(balance.minimum)
                },
                color = SlateSoft,
                style = Typography.bodySmall,
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.Center,
            )
        }

        if (payoutActionState != null && payoutActionState != "processing") {
            Spacer(modifier = Modifier.height(12.dp))
            ActionResultStrip(
                rawState = payoutActionState,
                onDismiss = onClearActionState,
            )
        }
    }
}

@Composable
private fun OpenRequestBlock(
    openRequest: PhotographerPayoutDto?,
    fallbackAmount: Double,
) {
    val status = openRequest?.status?.uppercase() ?: "PENDING_REVIEW"
    val tone = when (status) {
        "HELD" -> StatusTone.Danger
        "APPROVED" -> StatusTone.Approved
        else -> StatusTone.Warning
    }
    val statusLabel = when (status) {
        "HELD" -> "Payout · held"
        "APPROVED" -> "Payout · approved"
        else -> "Payout · pending"
    }

    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            StatusChip(text = statusLabel, tone = tone)
            Text(
                text = "₱%,.2f".format(openRequest?.amount ?: fallbackAmount),
                style = NumeralStyle.copy(fontSize = 15.sp, fontWeight = FontWeight.SemiBold),
                color = Ink,
            )
        }
        if (status == "HELD" && !openRequest?.holdReason.isNullOrBlank()) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Hold reason · ${openRequest?.holdReason}",
                color = ErrorRed,
                style = Typography.bodySmall,
            )
        }
    }
}

@Composable
private fun ActionResultStrip(
    rawState: String,
    onDismiss: () -> Unit,
) {
    val isError = rawState.startsWith("Error:")
    val tone = if (isError) StatusTone.Danger else StatusTone.Approved
    val toneColor = if (isError) ErrorRed else Fresh
    val message = rawState.removePrefix("Error: ").removePrefix("Success: ")

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(BadgeShape)
            .background(toneColor.copy(alpha = 0.08f))
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        StatusChip(
            text = if (isError) "Error" else "Submitted",
            tone = tone,
        )
        Text(
            text = message,
            color = Slate,
            style = Typography.bodySmall,
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 10.dp),
        )
        TextButton(onClick = onDismiss) {
            Text(
                text = "Dismiss",
                style = Typography.bodySmall,
                color = Ink,
                fontWeight = FontWeight.SemiBold,
            )
        }
    }
}

@Composable
private fun TransactionsLink(count: Int, onClick: () -> Unit) {
    TextButton(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
    ) {
        ArrowLabel(
            text = if (count == 0) "No transactions yet" else "View transactions ($count) →",
            color = if (count == 0) SlateSoft else Ink,
            style = Typography.bodyMedium,
            fontWeight = FontWeight.Medium,
        )
    }
}

@Composable
private fun TransactionsSheet(transactions: List<PhotographerTransactionDto>) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp)
            .padding(bottom = 24.dp),
    ) {
        Kicker(text = "Recent sales history", color = Slate)
        Spacer(modifier = Modifier.height(16.dp))
        if (transactions.isEmpty()) {
            Text(
                text = "No sales recorded yet. Keep uploading.",
                color = SlateSoft,
                style = Typography.bodyMedium,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 32.dp),
                textAlign = TextAlign.Center,
            )
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(420.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(transactions, key = { it.id }) { tx ->
                    TransactionRow(transaction = tx)
                }
            }
        }
    }
}

@Composable
private fun TransactionRow(transaction: PhotographerTransactionDto) {
    val buyerName = transaction.buyer.ifBlank { "Runner" }
    val initial = buyerName.firstOrNull()?.toString()?.uppercase() ?: "R"

    QpCard(modifier = Modifier.fillMaxWidth(), padding = 12.dp) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(CircleShape)
                    .background(BoneDeep),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = initial,
                    color = Slate,
                    style = Typography.labelMedium,
                    fontWeight = FontWeight.SemiBold,
                )
            }
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = buyerName,
                    color = Ink,
                    style = Typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = transaction.eventName ?: "Event archived",
                    color = Slate,
                    style = Typography.bodySmall,
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = formatIsoTimestamp(transaction.paidAt),
                    color = SlateSoft,
                    style = Typography.bodySmall,
                )
            }
            Text(
                text = "+₱%,.0f".format(transaction.amountKept),
                style = NumeralStyle.copy(fontSize = 15.sp, fontWeight = FontWeight.SemiBold),
                color = Ink,
            )
        }
    }
}

@Composable
private fun EarningsLoadingSkeleton() {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(24.dp),
    ) {
        LoadingSkeleton(modifier = Modifier.fillMaxWidth().height(56.dp))
        LoadingSkeleton(modifier = Modifier.fillMaxWidth().height(64.dp))
        Divider(thickness = 1.dp, color = Line)
        LoadingSkeleton(modifier = Modifier.fillMaxWidth().height(80.dp))
        Divider(thickness = 1.dp, color = Line)
        LoadingSkeleton(modifier = Modifier.fillMaxWidth().height(120.dp))
    }
}

private fun formatWeekLabel(weekOf: String): String = try {
    val parser = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
    val formatter = SimpleDateFormat("MMM d", Locale.getDefault())
    val date = parser.parse(weekOf) ?: Date()
    formatter.format(date)
} catch (e: Exception) {
    weekOf
}

private fun formatIsoTimestamp(isoString: String): String = try {
    val parser = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.getDefault())
    val formatter = SimpleDateFormat("MMM d, yyyy · h:mm a", Locale.getDefault())
    val date = parser.parse(isoString.substring(0, 19)) ?: Date()
    formatter.format(date)
} catch (e: Exception) {
    isoString.substringBefore("T")
}
