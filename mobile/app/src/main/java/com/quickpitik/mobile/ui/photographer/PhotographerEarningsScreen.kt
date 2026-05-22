package com.quickpitik.mobile.ui.photographer

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.data.remote.EarningsOverviewDto
import com.quickpitik.mobile.data.remote.PayoutBalanceDto
import com.quickpitik.mobile.data.remote.PhotographerTransactionDto
import com.quickpitik.mobile.ui.theme.*
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun PhotographerEarningsScreen(
    viewModel: PhotographerDashboardViewModel,
    modifier: Modifier = Modifier
) {
    val earningsState by viewModel.earningsUiState.collectAsState()
    val payoutActionState by viewModel.payoutActionState.collectAsState()

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Bone) // warm cream background
            .padding(16.dp)
    ) {
        // Header
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(bottom = 12.dp)
        ) {
            Column {
                Text(
                    text = "Earnings & Wallet",
                    color = Ink,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Track your platform payouts and digital receipts",
                    color = SlateSoft,
                    fontSize = 13.sp
                )
            }
        }

        // Main State Handling
        when (val state = earningsState) {
            is EarningsUiState.Loading -> {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(color = Fresh)
                }
            }
            is EarningsUiState.Error -> {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = state.message,
                            color = MaterialTheme.colorScheme.error,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.padding(16.dp)
                        )
                        Button(
                            onClick = { viewModel.fetchEarningsAndTransactions() },
                            colors = ButtonDefaults.buttonColors(containerColor = Fresh)
                        ) {
                            Text("Retry", color = Color.White)
                        }
                    }
                }
            }
            is EarningsUiState.Success -> {
                SuccessContent(
                    overview = state.overview,
                    balance = state.balance,
                    transactions = state.transactions,
                    payoutActionState = payoutActionState,
                    onRequestPayout = { viewModel.submitPayoutRequest() },
                    onClearActionState = { viewModel.clearPayoutActionState() },
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@Composable
private fun SuccessContent(
    overview: EarningsOverviewDto,
    balance: PayoutBalanceDto,
    transactions: List<PhotographerTransactionDto>,
    payoutActionState: String?,
    onRequestPayout: () -> Unit,
    onClearActionState: () -> Unit,
    modifier: Modifier = Modifier
) {
    LazyColumn(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        contentPadding = PaddingValues(vertical = 8.dp)
    ) {
        // 1. Metric Overview Cards
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                MetricCard(
                    title = "Lifetime Kept",
                    value = "₱%,.2f".format(overview.lifetimeKept),
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    title = "This Month",
                    value = "₱%,.2f".format(overview.thisMonth),
                    modifier = Modifier.weight(1f)
                )
            }
        }

        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                MetricCard(
                    title = "Pending Review",
                    value = "₱%,.2f".format(overview.payoutPending),
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    title = "Payout Cycle",
                    value = overview.payoutScheduledFor ?: "Weekly Review",
                    modifier = Modifier.weight(1f)
                )
            }
        }

        // 2. GCash Withdrawal Card
        item {
            Card(
                colors = CardDefaults.cardColors(containerColor = BoneDeep),
                border = BorderStroke(1.dp, Line),
                shape = RoundedCornerShape(24.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(20.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "GCASH WALLET BALANCE",
                            color = SlateSoft,
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        )
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .background(Fresh.copy(alpha = 0.1f), RoundedCornerShape(6.dp))
                                .padding(horizontal = 8.dp, vertical = 3.dp)
                        ) {
                            Text(
                                text = "INSTANT TRANSFER",
                                color = Fresh,
                                fontSize = 8.sp,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 0.5.sp
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "₱%,.2f".format(balance.unpaidBalance),
                        color = Fresh,
                        fontSize = 32.sp,
                        fontWeight = FontWeight.ExtraBold
                    )
                    Spacer(modifier = Modifier.height(14.dp))

                    val hasOpen = balance.hasOpenRequest || balance.openRequest != null
                    if (hasOpen) {
                        val openRequest = balance.openRequest
                        val status = openRequest?.status?.uppercase() ?: "PENDING_REVIEW"
                        val statusColor = when (status) {
                            "HELD" -> ErrorRed
                            "APPROVED" -> Fresh
                            else -> WarningOrange
                        }
                        
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(14.dp))
                                .background(Bone.copy(alpha = 0.9f))
                                .border(1.dp, Line, RoundedCornerShape(14.dp))
                                .padding(14.dp)
                        ) {
                            Column {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Box(
                                            modifier = Modifier
                                                .size(6.dp)
                                                .clip(CircleShape)
                                                .background(statusColor)
                                        )
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Text(
                                            text = "Payout: $status",
                                            color = statusColor,
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                    Text(
                                        text = "₱%,.2f".format(openRequest?.amount ?: balance.unpaidBalance),
                                        color = Ink,
                                        fontSize = 13.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                                if (status == "HELD" && !openRequest?.holdReason.isNullOrBlank()) {
                                    Spacer(modifier = Modifier.height(6.dp))
                                    Text(
                                        text = "Hold Reason: ${openRequest?.holdReason}",
                                        color = ErrorRed,
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.Medium
                                    )
                                }
                            }
                        }
                    } else {
                        // Allow withdrawal if balance >= minimum
                        val canWithdraw = balance.unpaidBalance >= balance.minimum
                        Button(
                            onClick = onRequestPayout,
                            enabled = canWithdraw && payoutActionState != "processing",
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Fresh,
                                contentColor = Color.White,
                                disabledContainerColor = Line,
                                disabledContentColor = SlateSoft
                            ),
                            shape = RoundedCornerShape(14.dp),
                            modifier = Modifier.fillMaxWidth().height(48.dp)
                        ) {
                            if (payoutActionState == "processing") {
                                CircularProgressIndicator(
                                    color = Color.White,
                                    modifier = Modifier.size(20.dp),
                                    strokeWidth = 2.dp
                                )
                            } else {
                                Text(
                                    text = "REQUEST PAYOUT TO GCASH",
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 12.sp,
                                    letterSpacing = 0.5.sp
                                )
                            }
                        }
                        
                        if (!canWithdraw) {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "Minimum withdrawal is ₱500.00",
                                color = SlateSoft,
                                fontSize = 11.sp,
                                modifier = Modifier.fillMaxWidth(),
                                textAlign = TextAlign.Center,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }

                    // Payout Action Result Alert Dialog / Snackbar Trigger
                    payoutActionState?.let { actionState ->
                        if (actionState != "processing") {
                            Spacer(modifier = Modifier.height(10.dp))
                            val isError = actionState.startsWith("Error:")
                            val alertColor = if (isError) ErrorRed else Fresh
                            
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(10.dp))
                                    .background(alertColor.copy(alpha = 0.1f))
                                    .border(1.dp, alertColor.copy(alpha = 0.3f), RoundedCornerShape(10.dp))
                                    .padding(10.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = actionState.replace("Error: ", "").replace("Success: ", ""),
                                    color = alertColor,
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold,
                                    modifier = Modifier.weight(1f)
                                )
                                TextButton(
                                    onClick = onClearActionState,
                                    contentPadding = PaddingValues(horizontal = 8.dp)
                                ) {
                                    Text("Dismiss", color = Ink, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Transactions Header
        item {
            Text(
                text = "Recent Sales History",
                color = Ink,
                fontSize = 17.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(top = 10.dp, bottom = 4.dp)
            )
        }

        // 4. Transactions List
        if (transactions.isEmpty()) {
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 32.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "No sales recorded yet. Keep uploading!",
                        color = SlateSoft,
                        fontSize = 14.sp
                    )
                }
            }
        } else {
            items(transactions) { tx ->
                TransactionItem(transaction = tx)
            }
        }
    }
}

@Composable
fun MetricCard(
    title: String,
    value: String,
    modifier: Modifier = Modifier
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        border = BorderStroke(1.dp, Line),
        shape = RoundedCornerShape(16.dp),
        modifier = modifier
    ) {
        Column(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth()
        ) {
            Text(
                text = title.uppercase(),
                color = SlateSoft,
                fontSize = 9.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.5.sp
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = value,
                color = Ink,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
fun TransactionItem(transaction: PhotographerTransactionDto) {
    val buyerName = transaction.buyer.ifBlank { "Runner" }
    val initial = buyerName.firstOrNull()?.toString()?.uppercase() ?: "R"

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp)
            .background(BoneDeep, RoundedCornerShape(16.dp))
            .border(1.dp, Line, RoundedCornerShape(16.dp))
            .padding(14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Initials Avatar Circle
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(CircleShape)
                .background(Fresh.copy(alpha = 0.15f)),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = initial,
                color = Fresh,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
        }
        
        Spacer(modifier = Modifier.width(12.dp))
        
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = "Purchased by $buyerName",
                color = Ink,
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = transaction.eventName ?: "Event archived",
                color = SlateSoft,
                fontSize = 12.sp
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = formatIsoTimestamp(transaction.paidAt),
                color = SlateSoft,
                fontSize = 10.sp
            )
        }
        
        Text(
            text = "+₱%,.2f".format(transaction.amountKept),
            color = Fresh,
            fontWeight = FontWeight.ExtraBold,
            fontSize = 15.sp
        )
    }
}

private fun formatIsoTimestamp(isoString: String): String {
    return try {
        val parser = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.getDefault())
        val formatter = SimpleDateFormat("MMM d, yyyy h:mm a", Locale.getDefault())
        val date = parser.parse(isoString.substring(0, 19)) ?: Date()
        formatter.format(date)
    } catch (e: Exception) {
        isoString.substringBefore("T")
    }
}
