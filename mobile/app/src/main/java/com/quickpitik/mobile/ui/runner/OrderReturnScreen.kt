package com.quickpitik.mobile.ui.runner

import android.content.Context
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.download.PhotoDownloader
import com.quickpitik.mobile.data.remote.OrderPhotoDetailDto
import com.quickpitik.mobile.data.remote.OrderDetailDto
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.NumeralStyle
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.QpHaptic
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.TileShape
import com.quickpitik.mobile.ui.theme.Typography
import com.quickpitik.mobile.ui.theme.rememberQpHaptic
import kotlinx.coroutines.launch
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

// /orders/return parity — mirrors website OrdersReturnPage. Polls the order
// until PayMongo's webhook flips status to PAID, then renders the editorial
// receipt: event meta, "All yours.", receipt block, photo cards with download
// CTAs. Mobile collapses the two-column layout into a single scroll, photo
// cards stacked under the receipt copy. Download saves the photo into the
// device's Pictures/QuickPitik gallery folder (PhotoDownloader).
@Composable
fun OrderReturnScreen(
    orderId: String,
    cartViewModel: CartViewModel,
    onNavigateToOrders: () -> Unit,
    onBrowseEvents: () -> Unit,
    onClose: () -> Unit,
) {
    val state by cartViewModel.orderReturnState.collectAsState()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }
    val hapticFire = rememberQpHaptic()
    var bulkBusy by remember { mutableStateOf(false) }

    LaunchedEffect(orderId) {
        cartViewModel.pollOrderReturn(orderId)
    }

    val downloadOne: (OrderPhotoDetailDto) -> Unit = { photo ->
        val url = photo.downloadUrl
        if (url.isNullOrBlank()) {
            scope.launch {
                snackbarHostState.showSnackbar("This photo isn't ready yet. Check back in a moment.")
            }
        } else {
            scope.launch {
                hapticFire(QpHaptic.CONFIRM)
                val filename = PhotoDownloader.buildFilename(photo.id, photo.bib)
                when (val res = PhotoDownloader.saveToGallery(context, url, filename)) {
                    is PhotoDownloader.Result.Saved -> {
                        snackbarHostState.showSnackbar("Saved to gallery · ${res.displayName}")
                    }
                    is PhotoDownloader.Result.Failed -> {
                        snackbarHostState.showSnackbar(res.message)
                    }
                }
            }
        }
    }

    val downloadAll: (List<OrderPhotoDetailDto>) -> Unit = { photos ->
        if (!bulkBusy) {
            scope.launch {
                bulkBusy = true
                hapticFire(QpHaptic.CONFIRM)
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

    Surface(modifier = Modifier.fillMaxSize(), color = Bone) {
        Box(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .statusBarsPadding()
                    .navigationBarsPadding(),
            ) {
                ReturnTopBar(onClose = onClose)
                when (val s = state) {
                    is OrderReturnState.Paid -> PaidBody(
                        detail = s.order,
                        bulkBusy = bulkBusy,
                        onDownloadOne = downloadOne,
                        onDownloadAll = { downloadAll(s.order.photos) },
                        onNavigateToOrders = onNavigateToOrders,
                        onBrowseEvents = onBrowseEvents,
                    )
                    is OrderReturnState.Polling -> PollingBody(attempt = s.attempt)
                    OrderReturnState.Timeout -> TimeoutBody(
                        orderId = orderId,
                        onNavigateToOrders = onNavigateToOrders,
                    )
                    is OrderReturnState.Failed -> FailedBody(
                        message = s.message,
                        onBrowseEvents = onBrowseEvents,
                    )
                    OrderReturnState.Idle -> PollingBody(attempt = 0)
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

@Composable
private fun ReturnTopBar(onClose: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        IconButton(onClick = onClose) {
            Icon(
                imageVector = Icons.Default.Close,
                contentDescription = "Close",
                tint = Ink,
            )
        }
    }
}

/* ───────── PAID ───────── */

@Composable
private fun PaidBody(
    detail: OrderDetailDto,
    bulkBusy: Boolean,
    onDownloadOne: (OrderPhotoDetailDto) -> Unit,
    onDownloadAll: () -> Unit,
    onNavigateToOrders: () -> Unit,
    onBrowseEvents: () -> Unit,
) {
    val photoCount = detail.photos.size
    val photoWord = if (photoCount == 1) "photo" else "photos"
    val eventName = detail.eventName ?: "QuickPitik"
    val recipientEmail = detail.recipientEmail.ifBlank { null }
    val reference = detail.id.take(8).uppercase(Locale.ROOT)
    val paidLabel = formatPaidAt(detail.paidAt)
    val showDownloadAll = photoCount >= 2 &&
        detail.photos.any { !it.downloadUrl.isNullOrBlank() }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 24.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp),
    ) {
        // Event meta + headline
        item {
            Column {
                Kicker(eventName)
                Spacer(modifier = Modifier.height(4.dp))
                Kicker(paidLabel, color = SlateSoft)
                Spacer(modifier = Modifier.height(20.dp))
                Kicker("── Payment received", color = Fresh)
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = "All yours.",
                    style = Typography.displayMedium,
                    color = Ink,
                )
                Spacer(modifier = Modifier.height(14.dp))
                Text(
                    text = buildString {
                        append(photoCount)
                        append(' ')
                        append(photoWord)
                        append(" from ")
                        append(eventName)
                        append(" for ₱")
                        append("%,.0f".format(detail.total))
                        append('.')
                    },
                    style = Typography.bodyLarge,
                    color = Slate,
                )
            }
        }

        // Receipt rows + email note
        item {
            Column {
                Divider(color = Line)
                Spacer(modifier = Modifier.height(16.dp))
                ReceiptRow(label = "Ref", value = reference)
                Spacer(modifier = Modifier.height(8.dp))
                ReceiptRow(label = "Paid", value = paidLabel)
                if (recipientEmail != null) {
                    Spacer(modifier = Modifier.height(8.dp))
                    ReceiptRow(label = "To", value = recipientEmail)
                }
                Spacer(modifier = Modifier.height(16.dp))
                Divider(color = Line)
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = if (recipientEmail != null) {
                        "We've also emailed these links to $recipientEmail. They work for 7 days."
                    } else {
                        "The download links above work for 7 days."
                    },
                    style = Typography.bodyMedium,
                    color = SlateSoft,
                )
            }
        }

        // Download-all CTA — only when there's >1 photo to bundle
        if (showDownloadAll) {
            item {
                PrimaryCta(
                    text = if (bulkBusy) "Saving to gallery…" else "↓ Save all $photoCount photos",
                    onClick = onDownloadAll,
                    enabled = !bulkBusy,
                    loading = bulkBusy,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        // Photo cards — each with their own per-photo Save CTA. When the master
        // "Save all" owns the fresh accent, demote per-photo to ghost so the
        // Quiet Studio "one fresh element per viewport" rule holds.
        itemsIndexed(detail.photos, key = { _, p -> p.id }) { index, photo ->
            PhotoReturnCard(
                photo = photo,
                index = index,
                useGhost = showDownloadAll,
                onDownload = { onDownloadOne(photo) },
            )
        }

        // Footer CTAs
        item {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                GhostCta(
                    text = "View your orders →",
                    onClick = onNavigateToOrders,
                    modifier = Modifier.fillMaxWidth(),
                )
                GhostCta(
                    text = "Browse more events →",
                    onClick = onBrowseEvents,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        item { Spacer(modifier = Modifier.height(24.dp)) }
    }
}

@Composable
private fun PhotoReturnCard(
    photo: OrderPhotoDetailDto,
    index: Int,
    useGhost: Boolean,
    onDownload: () -> Unit,
) {
    val previewSrc = photo.previewUrl ?: photo.thumbnailUrl
    val canDownload = !photo.downloadUrl.isNullOrBlank()
    val label = photo.bib?.let { "BIB $it" } ?: "Untagged"

    Column {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(4f / 5f)
                .clip(QpCardShape)
                .background(BoneDeep),
            contentAlignment = Alignment.Center,
        ) {
            if (previewSrc != null) {
                AsyncImage(
                    model = previewSrc,
                    contentDescription = label,
                    modifier = Modifier.fillMaxSize(),
                )
            } else {
                Text(
                    text = "QUICK\nPITIK",
                    style = Typography.labelSmall,
                    color = SlateSoft,
                    textAlign = TextAlign.Center,
                )
            }
            Box(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(12.dp)
                    .clip(TileShape)
                    .background(Bone.copy(alpha = 0.85f))
                    .padding(horizontal = 10.dp, vertical = 4.dp),
            ) {
                Kicker(label, color = SlateSoft)
            }
        }
        Spacer(modifier = Modifier.height(12.dp))
        val cta = "↓ Save photo${if (index > 0) " ${index + 1}" else ""}"
        if (!canDownload) {
            GhostCta(
                text = "Preparing · $label",
                onClick = {},
                enabled = false,
                modifier = Modifier.fillMaxWidth(),
            )
        } else if (useGhost) {
            GhostCta(
                text = cta,
                onClick = onDownload,
                modifier = Modifier.fillMaxWidth(),
            )
        } else {
            PrimaryCta(
                text = cta,
                onClick = onDownload,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

@Composable
private fun ReceiptRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Top,
    ) {
        Kicker(label, color = SlateSoft)
        Spacer(modifier = Modifier.width(24.dp))
        Text(
            text = value,
            style = NumeralStyle.copy(fontSize = 13.sp),
            color = Ink,
            textAlign = TextAlign.End,
        )
    }
}

/* ───────── POLLING / TIMEOUT / FAILED ───────── */

@Composable
private fun PollingBody(attempt: Int) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator(
            color = Fresh,
            strokeWidth = 2.dp,
            modifier = Modifier.size(40.dp),
        )
        Spacer(modifier = Modifier.height(24.dp))
        Kicker(
            text = if (attempt > 0) "Confirming · attempt $attempt" else "Confirming",
            color = SlateSoft,
        )
        Spacer(modifier = Modifier.height(10.dp))
        Text(
            text = "Sealing your photos.",
            style = Typography.displaySmall,
            color = Ink,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(10.dp))
        Text(
            text = "We're waiting on PayMongo to confirm the charge. Usually a few seconds — keep this screen open.",
            style = Typography.bodyMedium,
            color = SlateSoft,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
private fun TimeoutBody(orderId: String, onNavigateToOrders: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Kicker("Still processing", color = SlateSoft)
        Spacer(modifier = Modifier.height(10.dp))
        Text(
            text = "Your bank is taking a moment.",
            style = Typography.displaySmall,
            color = Ink,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "PayMongo confirmed your payment, but the receipt hasn't reached us yet. Check your email in a minute — we'll send the download links the moment it lands.",
            style = Typography.bodyMedium,
            color = SlateSoft,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(24.dp))
        PrimaryCta(text = "View your orders →", onClick = onNavigateToOrders)
        Spacer(modifier = Modifier.height(20.dp))
        Kicker(
            text = "Reference · ${orderId.take(8).uppercase(Locale.ROOT)}",
            color = SlateSoft,
        )
    }
}

@Composable
private fun FailedBody(message: String, onBrowseEvents: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Kicker("Something went sideways", color = ErrorRed)
        Spacer(modifier = Modifier.height(10.dp))
        Text(
            text = "We couldn't confirm this order.",
            style = Typography.displaySmall,
            color = Ink,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = message.ifBlank {
                "If you were charged, the receipt will still arrive by email. Otherwise nothing was taken — try again any time."
            },
            style = Typography.bodyMedium,
            color = SlateSoft,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(24.dp))
        PrimaryCta(text = "Browse events →", onClick = onBrowseEvents)
    }
}

/* ───────── helpers ───────── */

// "MAY 27 2026 · 3:42 PM PHT" — mirrors the website formatPaidAt.
private fun formatPaidAt(iso: String?): String {
    if (iso.isNullOrBlank()) return "—"
    return try {
        val zoned = OffsetDateTime.parse(iso).atZoneSameInstant(MANILA)
        val date = zoned.format(DATE_FMT).uppercase(Locale.ROOT)
        val time = zoned.format(TIME_FMT).uppercase(Locale.ROOT)
        "$date · $time PHT"
    } catch (_: Exception) {
        iso
    }
}

private val MANILA: ZoneId = ZoneId.of("Asia/Manila")
private val DATE_FMT: DateTimeFormatter = DateTimeFormatter.ofPattern("MMM d yyyy", Locale.ENGLISH)
private val TIME_FMT: DateTimeFormatter = DateTimeFormatter.ofPattern("h:mm a", Locale.ENGLISH)
