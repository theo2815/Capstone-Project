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
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
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
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.*
import kotlinx.coroutines.launch
import java.time.OffsetDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

// Receipt paging, mirroring the website's PAGE_SIZE.RECEIPT_INITIAL / +10.
private const val RECEIPT_INITIAL = 10
private const val RECEIPT_PAGE = 10

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OrdersScreen(
    viewModel: CartViewModel,
    initialOrderId: String? = null,
    onLogout: () -> Unit = {}
) {
    val ordersState by viewModel.ordersState.collectAsState()
    val orderDetailState by viewModel.orderDetailState.collectAsState()
    val refundAction by viewModel.refundActionState.collectAsState()

    // rememberSaveable: rotating mid-receipt used to bounce the runner back to
    // the list top with the detail closed.
    var selectedOrderId by rememberSaveable { mutableStateOf(initialOrderId) }
    // How many receipts are rendered; grows by RECEIPT_PAGE on "LOAD MORE".
    var receiptLimit by rememberSaveable { mutableStateOf(RECEIPT_INITIAL) }
    // Index into the open order's photos while the owned lightbox is showing.
    var ownedPreviewIndex by rememberSaveable { mutableStateOf<Int?>(null) }
    var showRefundDialog by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }
    var bulkBusy by remember { mutableStateOf(false) }
    // Pull-to-refresh (Mobile Design skill) — spinner settles when the fetch
    // Job completes.
    var ordersRefreshing by remember { mutableStateOf(false) }

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
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(top = 24.dp)
        ) {
            if (selectedOrderId == null) {
                // Main Orders List View
                RunnerTopBar(
                    kicker = "ORDER HISTORY",
                    onLogout = onLogout
                )
                Spacer(modifier = Modifier.height(24.dp))

                when (val state = ordersState) {
                    is OrdersState.Loading -> {
                        Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                            CircularProgressIndicator(color = Fresh)
                        }
                    }
                    is OrdersState.Error -> {
                        Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                            ErrorView(
                                message = state.message,
                                title = "Couldn't load your orders",
                                onRetry = { viewModel.fetchOrders() },
                            )
                        }
                    }
                    is OrdersState.Success -> {
                        // Every order, PENDING included — website parity. The web's
                        // /orders lists an abandoned PayMongo session as a row with
                        // its status; hiding it here also made this screen's spend
                        // math disagree with the profile race log's (which never
                        // filtered), so the same runner saw two lifetime totals.
                        val visibleOrders = state.orders
                        if (visibleOrders.isEmpty()) {
                            PullToRefreshBox(
                                isRefreshing = ordersRefreshing,
                                onRefresh = {
                                    ordersRefreshing = true
                                    scope.launch {
                                        viewModel.fetchOrders().join()
                                        ordersRefreshing = false
                                    }
                                },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .weight(1f),
                            ) {
                            LazyColumn(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(horizontal = 24.dp),
                                verticalArrangement = Arrangement.spacedBy(16.dp),
                                contentPadding = PaddingValues(bottom = 24.dp)
                            ) {
                                item {
                                    Column(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(vertical = 8.dp),
                                        verticalArrangement = Arrangement.spacedBy(16.dp)
                                    ) {
                                        Kicker("01  SPEND")
                                        Text(
                                            text = "Your purchase totals will land here once you keep your first photo.",
                                            style = Typography.bodyMedium,
                                            color = Slate
                                        )
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Divider(color = Line, thickness = 1.dp)
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Kicker("02  RECEIPTS")
                                        Text(
                                            text = "No receipts yet. Complete your first checkout to view receipts here.",
                                            style = Typography.bodyMedium,
                                            color = Slate
                                        )
                                    }
                                }
                            }
                            }
                        } else {
                            val spendStats = remember(visibleOrders) { computeSpendStats(visibleOrders) }
                            val pagedOrders = visibleOrders.take(receiptLimit)
                            PullToRefreshBox(
                                isRefreshing = ordersRefreshing,
                                onRefresh = {
                                    ordersRefreshing = true
                                    scope.launch {
                                        viewModel.fetchOrders().join()
                                        ordersRefreshing = false
                                    }
                                },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .weight(1f),
                            ) {
                            LazyColumn(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(horizontal = 24.dp),
                                contentPadding = PaddingValues(bottom = 24.dp)
                            ) {
                                item { SpendSection(spendStats) }
                                item {
                                    Spacer(modifier = Modifier.height(16.dp))
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(bottom = 8.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Kicker("02  RECEIPTS")
                                        Kicker(
                                            text = "${visibleOrders.size} ${if (visibleOrders.size == 1) "RECEIPT" else "RECEIPTS"}",
                                            color = SlateSoft
                                        )
                                    }
                                }
                                items(pagedOrders, key = { it.id }) { order ->
                                    ReceiptRowItem(
                                        order = order,
                                        onClick = { selectedOrderId = order.id }
                                    )
                                    Divider(color = Line, thickness = 1.dp)
                                }
                                if (visibleOrders.size > pagedOrders.size) {
                                    item {
                                        Spacer(modifier = Modifier.height(16.dp))
                                        GhostCta(
                                            text = "LOAD MORE",
                                            onClick = { receiptLimit += RECEIPT_PAGE },
                                            modifier = Modifier.fillMaxWidth(),
                                        )
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
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp)
                        .padding(horizontal = 24.dp),
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
                                ErrorView(
                                    message = detailState.message,
                                    title = "Couldn't load this order",
                                    onRetry = { selectedOrderId?.let { viewModel.fetchOrderDetail(it) } },
                                )
                                GhostCta(
                                    text = "Back to list",
                                    onClick = { selectedOrderId = null },
                                )
                            }
                        }
                    }
                    is OrderDetailState.Success -> {
                        val order = detailState.order
                        LazyColumn(
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f)
                                .padding(horizontal = 24.dp),
                            verticalArrangement = Arrangement.spacedBy(16.dp),
                            contentPadding = PaddingValues(bottom = 24.dp)
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
                                        // Tap opens the owned lightbox — full-size,
                                        // swipeable, un-watermarked. The row's own
                                        // Download button still works for a direct save.
                                        onClick = {
                                            ownedPreviewIndex = order.photos.indexOfFirst { it.id == photo.id }
                                                .takeIf { it >= 0 }
                                        },
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
                                                        model = RetrofitClient.resolveImageUrl(photo.thumbnailUrl),
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

    // Owned-photo lightbox. Reuses the shared PhotoPreview primitive in its
    // Owned mode rather than a bespoke dialog — same pager, same chrome the
    // runner already knows from browsing.
    val openOrder = (orderDetailState as? OrderDetailState.Success)?.order
    if (openOrder != null) {
        ownedPreviewIndex?.let { index ->
            PhotoPreview(
                photos = openOrder.photos.map { it.toOwnedPreviewData(openOrder.eventName) },
                currentIndex = index,
                mode = PhotoPreviewMode.Owned,
                onClose = { ownedPreviewIndex = null },
                onIndexChange = { ownedPreviewIndex = it },
                onDownload = { data ->
                    openOrder.photos.firstOrNull { it.id == data.id }?.let(downloadOne)
                },
            )
        }
    }
}

// previewUrl is already the CLEAN original for an order the caller owns
// (backend OrderService.previewUrlOf serves photo.s3Key — the G-2 fix), so no
// cleanUrl field is needed here. thumbnailUrl is the watermarked fallback.
private fun OrderPhotoDetailDto.toOwnedPreviewData(eventName: String?): PhotoPreviewData =
    PhotoPreviewData(
        id = id,
        price = 0.0,
        imageUrl = previewUrl ?: thumbnailUrl,
        eventName = eventName,
    )

// Mirrors website /orders SpendSlab — "Lifetime totals" snapshot. Three stats
// across the top of the list: total spent (fresh accent), order count, photos
// kept. "Since {month-year}" derived from the earliest paidAt, mirroring the
// website's `computeSpendStats`. Totals include every order (PAID + PENDING),
// same as the website — PENDING rows are abandoned PayMongo sessions and
// historically rare.
@Composable
private fun SpendSection(stats: SpendStats) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Kicker("01  SPEND")

        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = String.format(Locale.US, "₱%,.2f", stats.total),
                style = NumeralStyle.copy(fontSize = 36.sp, fontWeight = FontWeight.ExtraBold),
                color = Fresh
            )
            Kicker(text = "SPENT", color = SlateSoft)
        }

        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = stats.orderCount.toString(),
                style = NumeralStyle.copy(fontSize = 28.sp, fontWeight = FontWeight.Bold),
                color = Ink
            )
            Kicker(text = if (stats.orderCount == 1) "ORDER" else "ORDERS", color = SlateSoft)
        }

        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = stats.photoCount.toString(),
                style = NumeralStyle.copy(fontSize = 28.sp, fontWeight = FontWeight.Bold),
                color = Ink
            )
            Kicker(text = if (stats.photoCount == 1) "PHOTO KEPT" else "PHOTOS KEPT", color = SlateSoft)
        }

        if (stats.firstPurchase != null) {
            Spacer(modifier = Modifier.height(4.dp))
            Kicker(text = "SINCE ${stats.firstPurchase.uppercase()}", color = SlateSoft)
        }

        Spacer(modifier = Modifier.height(8.dp))
        Divider(color = Line, thickness = 1.dp)
    }
}

@Composable
private fun ReceiptRowItem(
    order: OrderListItemDto,
    onClick: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        // Date kicker (e.g. MAY 28 · 2026)
        Kicker(
            text = formatReceiptDate(order.paidAt),
            color = SlateSoft
        )

        // Event Name
        Text(
            text = order.eventName ?: "Event Photos",
            style = Typography.titleMedium.copy(fontWeight = FontWeight.Bold),
            color = Ink
        )

        // Metadata line
        val photoCountLabel = if (order.photoIds.size == 1) "1 photo" else "${order.photoIds.size} photos"
        val paymentMethod = labelForPaymentMethod(order.paymentMethod)
        val orderRef = order.id.take(36)
        Text(
            text = "$photoCountLabel · $paymentMethod · $orderRef",
            style = Typography.bodySmall,
            color = Slate
        )

        // Refund status chip (if any)
        val rollup = computeRefundRollup(
            photoCount = order.photoIds.size,
            disputes = order.disputes
        )
        RefundStatusChip(
            rollup = rollup,
            photoCount = order.photoIds.size
        )

        Spacer(modifier = Modifier.height(4.dp))

        // Price & "View photos →"
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = String.format(Locale.US, "₱%,.2f", order.total),
                style = NumeralStyle.copy(fontSize = 20.sp, fontWeight = FontWeight.Bold),
                color = Ink
            )

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = "View photos",
                    style = Typography.bodySmall.copy(fontWeight = FontWeight.Medium),
                    color = Ink
                )
                Icon(
                    imageVector = Icons.Default.KeyboardArrowRight,
                    contentDescription = "View photos",
                    tint = Ink,
                    modifier = Modifier.size(16.dp)
                )
            }
        }
    }
}

private fun labelForPaymentMethod(method: String?): String {
    if (method.isNullOrBlank()) return "Online"
    return when (method.lowercase()) {
        "gcash" -> "GCash"
        "card" -> "Card"
        "paymaya", "maya" -> "PayMaya"
        "grabpay", "grab_pay" -> "GrabPay"
        else -> method.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.US) else it.toString() }
    }
}

private val RECEIPT_DATE_FMT: DateTimeFormatter =
    DateTimeFormatter.ofPattern("MMM dd · yyyy", Locale.US)

private fun formatReceiptDate(iso: String?): String {
    val dt = parsePaidAt(iso) ?: return "RECENT"
    return dt.format(RECEIPT_DATE_FMT).uppercase()
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
