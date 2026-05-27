package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.download.PhotoDownloader
import com.quickpitik.mobile.data.remote.OrderListItemDto
import com.quickpitik.mobile.data.remote.OrderPhotoDetailDto
import com.quickpitik.mobile.ui.theme.*
import kotlinx.coroutines.launch
import java.time.OffsetDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OrdersScreen(
    viewModel: CartViewModel,
    onNavigateBack: () -> Unit
) {
    val ordersState by viewModel.ordersState.collectAsState()
    val orderDetailState by viewModel.orderDetailState.collectAsState()
    val refundAction by viewModel.refundActionState.collectAsState()

    var selectedOrderId by remember { mutableStateOf<String?>(null) }
    var showRefundDialog by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }
    var bulkBusy by remember { mutableStateOf(false) }

    // Per-photo + bundle download — mirrors website /orders triggerDownload
    // (anchor + Content-Disposition: attachment), but lands the JPEG straight
    // in the device's Pictures/QuickPitik gallery folder via MediaStore.
    // Same pattern as OrderReturnScreen's onDownloadOne/onDownloadAll.
    val downloadOne: (OrderPhotoDetailDto) -> Unit = { photo ->
        val url = photo.downloadUrl
        if (url.isNullOrBlank()) {
            scope.launch {
                snackbarHostState.showSnackbar("This photo isn't ready yet. Check back in a moment.")
            }
        } else {
            scope.launch {
                val filename = PhotoDownloader.buildFilename(photo.id, photo.bib)
                when (val res = PhotoDownloader.saveToGallery(context, url, filename)) {
                    is PhotoDownloader.Result.Saved ->
                        snackbarHostState.showSnackbar("Saved to gallery · ${res.displayName}")
                    is PhotoDownloader.Result.Failed ->
                        snackbarHostState.showSnackbar(res.message)
                }
            }
        }
    }

    val downloadAll: (List<OrderPhotoDetailDto>) -> Unit = { photos ->
        if (!bulkBusy) {
            scope.launch {
                bulkBusy = true
                var saved = 0
                var failed = 0
                for (photo in photos) {
                    val url = photo.downloadUrl
                    if (url.isNullOrBlank()) {
                        failed++
                        continue
                    }
                    val filename = PhotoDownloader.buildFilename(photo.id, photo.bib)
                    when (PhotoDownloader.saveToGallery(context, url, filename)) {
                        is PhotoDownloader.Result.Saved -> saved++
                        is PhotoDownloader.Result.Failed -> failed++
                    }
                }
                val summary = buildString {
                    append("Saved $saved of ${photos.size} to gallery.")
                    if (failed > 0) append(" $failed couldn't be saved.")
                }
                snackbarHostState.showSnackbar(summary)
                bulkBusy = false
            }
        }
    }

    LaunchedEffect(selectedOrderId) {
        if (selectedOrderId != null) {
            viewModel.resetRefundActionState()
            viewModel.fetchOrderDetail(selectedOrderId!!)
        } else {
            viewModel.fetchOrders()
            viewModel.resetOrderDetailState()
            viewModel.resetRefundActionState()
        }
    }

    // Close the dialog once the server accepts the request; the banner + refreshed
    // timeline then render in place behind it.
    LaunchedEffect(refundAction) {
        if (refundAction is RefundActionState.Success) {
            showRefundDialog = false
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Bone
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp)
                .statusBarsPadding()
                .navigationBarsPadding()
        ) {
            if (selectedOrderId == null) {
                // Main Orders List View
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(
                        onClick = onNavigateBack,
                        colors = IconButtonDefaults.iconButtonColors(containerColor = BoneDeep)
                    ) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = Ink)
                    }
                    Spacer(modifier = Modifier.width(16.dp))
                    Column {
                        Kicker("Marketplace")
                        Text(
                            text = "Order history",
                            style = Typography.headlineSmall,
                            color = Ink
                        )
                    }
                }
                Spacer(modifier = Modifier.height(24.dp))

                when (val state = ordersState) {
                    is OrdersState.Loading -> {
                        Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                            CircularProgressIndicator(color = Fresh)
                        }
                    }
                    is OrdersState.Error -> {
                        Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(24.dp)) {
                                Icon(Icons.Default.Warning, contentDescription = "Error", tint = ErrorRed, modifier = Modifier.size(48.dp))
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(state.message, color = ErrorRed, style = Typography.bodyMedium, textAlign = TextAlign.Center)
                            }
                        }
                    }
                    is OrdersState.Success -> {
                        // Order history = things the runner actually paid for. PENDING rows
                        // (abandoned PayMongo sessions) come back from the backend in the
                        // same list, but they don't belong on a "history" surface — filter
                        // them out before they reach the spend math or the LazyColumn.
                        val visibleOrders = remember(state.orders) {
                            state.orders.filter { it.status == "PAID" || it.status == "FULFILLED" }
                        }
                        if (visibleOrders.isEmpty()) {
                            Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(32.dp)) {
                                    Text("No Orders Found", style = Typography.titleMedium, fontWeight = FontWeight.Bold, color = Ink)
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text(
                                        text = "Any photo checkouts you complete on this account will be listed here for digital high-res downloading.",
                                        style = Typography.bodyMedium,
                                        color = SlateSoft,
                                        textAlign = TextAlign.Center
                                    )
                                }
                            }
                        } else {
                            val spendStats = remember(visibleOrders) { computeSpendStats(visibleOrders) }
                            LazyColumn(
                                modifier = Modifier.fillMaxWidth().weight(1f),
                                verticalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                item { SpendSection(spendStats) }
                                items(visibleOrders, key = { it.id }) { order ->
                                    Card(
                                        onClick = { selectedOrderId = order.id },
                                        colors = CardDefaults.cardColors(containerColor = BoneDeep),
                                        border = BorderStroke(1.dp, Line),
                                        shape = QpCardShape,
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Row(
                                            modifier = Modifier.padding(16.dp),
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.SpaceBetween
                                        ) {
                                            Column(modifier = Modifier.weight(1f)) {
                                                Text(
                                                    text = order.eventName ?: "Marathon Event",
                                                    style = Typography.titleMedium,
                                                    color = Ink
                                                )
                                                Text(
                                                    text = "${order.photoIds.size} photos · ${order.paymentMethod}",
                                                    style = Typography.bodySmall,
                                                    color = SlateSoft
                                                )
                                                Spacer(modifier = Modifier.height(6.dp))
                                                Text(
                                                    text = String.format("₱%,.2f", order.total),
                                                    style = NumeralStyle.copy(fontSize = 16.sp),
                                                    color = Fresh
                                                )
                                            }
                                            // No payment-status chip — mirrors website /orders which never
                                            // renders PAID/PENDING on the receipt row. PENDING orders (abandoned
                                            // PayMongo sessions) still appear in the list because the backend
                                            // returns every order; the chevron alone signals tap-for-detail.
                                            Icon(Icons.Default.KeyboardArrowRight, contentDescription = "Detail", tint = Ink)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // Order Detail & High-Res Download View
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(
                        onClick = { selectedOrderId = null },
                        colors = IconButtonDefaults.iconButtonColors(containerColor = BoneDeep)
                    ) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back to List", tint = Ink)
                    }
                    Spacer(modifier = Modifier.width(16.dp))
                    Column {
                        Kicker("Order details")
                        Text(
                            text = "Download panel",
                            style = Typography.headlineSmall,
                            color = Ink
                        )
                    }
                }
                Spacer(modifier = Modifier.height(24.dp))

                when (val detailState = orderDetailState) {
                    is OrderDetailState.Loading -> {
                        Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                            CircularProgressIndicator(color = Fresh)
                        }
                    }
                    is OrderDetailState.Error -> {
                        Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Icon(Icons.Default.Warning, contentDescription = "Error", tint = ErrorRed, modifier = Modifier.size(48.dp))
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(detailState.message, color = ErrorRed, style = Typography.bodyMedium, textAlign = TextAlign.Center)
                                Spacer(modifier = Modifier.height(16.dp))
                                Button(onClick = { selectedOrderId = null }, colors = ButtonDefaults.buttonColors(containerColor = Line, contentColor = Ink)) {
                                    Text("BACK TO LIST")
                                }
                            }
                        }
                    }
                    is OrderDetailState.Success -> {
                        val order = detailState.order
                        LazyColumn(
                            modifier = Modifier.fillMaxWidth().weight(1f),
                            verticalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            // Order summary header card + bundle download
                            item {
                                Card(
                                    colors = CardDefaults.cardColors(containerColor = SurfaceWhite),
                                    border = BorderStroke(1.dp, Line),
                                    shape = RoundedCornerShape(16.dp),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Column(modifier = Modifier.padding(16.dp)) {
                                        Text(
                                            text = order.eventName ?: "Event Photos Bundle",
                                            style = Typography.titleMedium,
                                            fontWeight = FontWeight.Bold,
                                            color = Ink
                                        )
                                        Text(
                                            text = "Status: ${order.status} | Recipient: ${order.recipientEmail}",
                                            style = Typography.bodySmall,
                                            color = SlateSoft
                                        )
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Text(
                                                text = String.format("Total paid: ₱%,.2f", order.total),
                                                style = Typography.titleSmall,
                                                fontWeight = FontWeight.Bold,
                                                color = Fresh
                                            )
                                            // Mobile diverges intentionally from website's ZIP bundle: photos
                                            // belong in the gallery, not a downloads folder. Iterate every photo
                                            // and save each as a JPEG via MediaStore. Same pattern as the
                                            // /orders/return Save-all CTA.
                                            if (order.photos.any { !it.downloadUrl.isNullOrBlank() }) {
                                                Button(
                                                    onClick = { downloadAll(order.photos) },
                                                    enabled = !bulkBusy,
                                                    colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                                                    shape = RoundedCornerShape(8.dp)
                                                ) {
                                                    Text(
                                                        text = if (bulkBusy) "SAVING…" else "SAVE ALL",
                                                        style = Typography.labelSmall,
                                                    )
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Success / error feedback from the latest refund action
                            (refundAction as? RefundActionState.Success)?.let {
                                item { RefundStatusBanner(it.message, isError = false) }
                            }
                            (refundAction as? RefundActionState.Error)?.let {
                                item { RefundStatusBanner(it.message, isError = true) }
                            }

                            // Refund request / cancel actions
                            item {
                                RefundActionsRow(
                                    order = order,
                                    submitting = refundAction is RefundActionState.Submitting,
                                    onRequest = {
                                        viewModel.resetRefundActionState()
                                        showRefundDialog = true
                                    },
                                    onCancel = { disputeId -> viewModel.withdrawDispute(order.id, disputeId) }
                                )
                            }

                            item { Kicker("Purchased photos") }

                            items(order.photos, key = { it.id }) { photo ->
                                    Card(
                                        colors = CardDefaults.cardColors(containerColor = BoneDeep),
                                        border = BorderStroke(1.dp, Line),
                                        shape = QpCardShape,
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Row(
                                            modifier = Modifier.padding(10.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Box(
                                                modifier = Modifier
                                                    .size(60.dp)
                                                    .clip(RoundedCornerShape(8.dp))
                                                    .background(Line),
                                                contentAlignment = Alignment.Center
                                            ) {
                                                if (photo.thumbnailUrl != null) {
                                                    AsyncImage(
                                                        model = photo.thumbnailUrl,
                                                        contentDescription = "Purchased photo thumbnail",
                                                        modifier = Modifier.fillMaxSize()
                                                    )
                                                } else {
                                                    Icon(
                                                        imageVector = Icons.Default.CheckCircle,
                                                        contentDescription = "Unlocked",
                                                        tint = Fresh
                                                    )
                                                }
                                            }
                                            Spacer(modifier = Modifier.weight(1f))

                                            // Download high res action button
                                            if (photo.downloadUrl != null) {
                                                Button(
                                                    onClick = { downloadOne(photo) },
                                                    colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                                                    shape = RoundedCornerShape(8.dp)
                                                ) {
                                                    Text("DOWNLOAD", style = Typography.labelSmall)
                                                }
                                            } else {
                                                Text(
                                                    text = "Unlocking...",
                                                    style = Typography.bodySmall,
                                                    color = SlateSoft,
                                                    modifier = Modifier.padding(end = 8.dp)
                                                )
                                            }
                                        }
                                    }
                                }

                            // Refund history timeline (per-dispute lifecycle)
                            if (order.disputes.isNotEmpty()) {
                                item { RefundTimeline(order.disputes) }
                            }
                        }

                        if (showRefundDialog) {
                            RefundRequestDialog(
                                order = order,
                                submitting = refundAction is RefundActionState.Submitting,
                                onDismiss = { showRefundDialog = false },
                                onSubmit = { photoIds, reason, note ->
                                    viewModel.submitRefund(order.id, photoIds, reason, note)
                                }
                            )
                        }
                    }
                    else -> {}
                }
            }
        }
            SnackbarHost(
                hostState = snackbarHostState,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .navigationBarsPadding()
                    .padding(horizontal = 16.dp, vertical = 12.dp),
            ) { data ->
                Snackbar(
                    snackbarData = data,
                    containerColor = Ink,
                    contentColor = Bone,
                    shape = QpCardShape,
                )
            }
        }
    }
}

// Mirrors website /orders SpendSlab — "Lifetime totals" snapshot. Three stats
// across the top of the list: total spent (fresh accent), order count, photos
// kept. "Since {month-year}" derived from the earliest paidAt, mirroring the
// website's `computeSpendStats`. Totals include every order (PAID + PENDING),
// same as the website — PENDING rows are abandoned PayMongo sessions and
// historically rare.
@Composable
private fun SpendSection(stats: SpendStats) {
    Card(
        colors = CardDefaults.cardColors(containerColor = SurfaceWhite),
        border = BorderStroke(1.dp, Line),
        shape = QpCardShape,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(horizontal = 20.dp, vertical = 18.dp)) {
            Kicker("Lifetime totals")
            Spacer(modifier = Modifier.height(14.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.Top
            ) {
                SpendStatCell(
                    value = String.format(Locale.ENGLISH, "₱%,d", stats.total.toLong()),
                    label = "spent",
                    accent = true,
                    modifier = Modifier.weight(1f)
                )
                Divider(
                    color = Line,
                    modifier = Modifier
                        .width(1.dp)
                        .height(56.dp)
                )
                SpendStatCell(
                    value = stats.orderCount.toString(),
                    label = if (stats.orderCount == 1) "order" else "orders",
                    accent = false,
                    modifier = Modifier.weight(1f)
                )
                Divider(
                    color = Line,
                    modifier = Modifier
                        .width(1.dp)
                        .height(56.dp)
                )
                SpendStatCell(
                    value = stats.photoCount.toString(),
                    label = if (stats.photoCount == 1) "photo kept" else "photos kept",
                    accent = false,
                    modifier = Modifier.weight(1f)
                )
            }
            if (stats.firstPurchase != null) {
                Spacer(modifier = Modifier.height(14.dp))
                Kicker(text = "Since ${stats.firstPurchase}", color = SlateSoft)
            }
        }
    }
}

@Composable
private fun SpendStatCell(
    value: String,
    label: String,
    accent: Boolean,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.padding(horizontal = 8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Text(
            text = value,
            style = NumeralStyle.copy(fontSize = 22.sp),
            color = if (accent) Fresh else Ink
        )
        Kicker(label)
    }
}

private data class SpendStats(
    val total: Double,
    val orderCount: Int,
    val photoCount: Int,
    val firstPurchase: String?
)

private fun computeSpendStats(orders: List<OrderListItemDto>): SpendStats {
    if (orders.isEmpty()) {
        return SpendStats(0.0, 0, 0, null)
    }
    var total = 0.0
    var photoCount = 0
    var earliest: OffsetDateTime? = null
    for (o in orders) {
        total += o.total
        photoCount += o.photoIds.size
        val paid = parsePaidAt(o.paidAt)
        if (paid != null && (earliest == null || paid.isBefore(earliest))) {
            earliest = paid
        }
    }
    return SpendStats(
        total = total,
        orderCount = orders.size,
        photoCount = photoCount,
        firstPurchase = earliest?.format(SPEND_MONTH_FMT)
    )
}

private fun parsePaidAt(iso: String?): OffsetDateTime? {
    if (iso.isNullOrBlank()) return null
    return try {
        OffsetDateTime.parse(iso)
    } catch (_: Exception) {
        null
    }
}

private val SPEND_MONTH_FMT: DateTimeFormatter =
    DateTimeFormatter.ofPattern("MMM yyyy", Locale.ENGLISH)
