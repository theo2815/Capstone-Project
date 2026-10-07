package com.quickpitik.mobile.ui.photographer

import android.widget.Toast
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.quickpitik.mobile.data.download.PhotoDownloader
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.PhotographerLibraryPhotoDto
import com.quickpitik.mobile.data.remote.PhotographerPurchaseDto
import com.quickpitik.mobile.data.remote.PhotographerTransactionDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.common.PhotoPreview
import com.quickpitik.mobile.ui.common.PhotoPreviewData
import com.quickpitik.mobile.ui.common.PhotoPreviewMode
import com.quickpitik.mobile.ui.common.formatManila
import com.quickpitik.mobile.ui.common.formatOrderRef
import com.quickpitik.mobile.ui.common.peso
import com.quickpitik.mobile.ui.common.toOwnerPreviewData
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.Typography
import kotlinx.coroutines.launch
import kotlin.math.abs

/** Seller purchase context inside the existing owner preview shell. */
@Composable
fun PhotographerPurchasePreview(
    viewModel: PhotographerPurchasesViewModel,
    photos: List<PhotographerLibraryPhotoDto> = emptyList(),
    transaction: PhotographerTransactionDto? = null,
    onClose: () -> Unit,
    onOpenEvent: (String) -> Unit,
) {
    val state by viewModel.previewState.collectAsState()
    val photoId = state.photoId ?: return
    val context = LocalContext.current
    val session = remember(context) { SessionManager.getInstance(context) }
    val scope = rememberCoroutineScope()
    var downloadingPhotoId by remember { mutableStateOf<String?>(null) }
    var downloadErrorPhotoId by remember { mutableStateOf<String?>(null) }
    val knownPhoto = state.photo ?: photos.firstOrNull { it.id == photoId }
    val gallery = if (photos.any { it.id == photoId }) photos else listOfNotNull(knownPhoto)
    val previewPhotos = if (gallery.isNotEmpty()) gallery.map { photo ->
        (if (photo.id == photoId) state.photo ?: photo else photo).toOwnerPreviewData(state.event?.name ?: transaction?.eventName)
    } else listOf(PhotoPreviewData(photoId, 0.0, transaction?.thumbnailUrl, transaction?.eventName))
    val index = previewPhotos.indexOfFirst { it.id == photoId }.coerceAtLeast(0)

    fun close() {
        viewModel.closePreview()
        onClose()
    }

    PhotoPreview(
        photos = previewPhotos,
        currentIndex = index,
        onClose = ::close,
        onIndexChange = { next -> viewModel.openPurchase(previewPhotos[next].id) },
        mode = PhotoPreviewMode.OwnerReview,
        canDownload = { photo -> state.photo?.id == photo.id && state.photo?.deletionStartedAt == null && downloadingPhotoId != photo.id },
        onDownload = { photo ->
            val token = session.getAccessToken()
            if (downloadingPhotoId != photo.id && token != null && state.photo?.id == photo.id && state.photo?.deletionStartedAt == null) {
                downloadingPhotoId = photo.id
                downloadErrorPhotoId = null
                val generation = session.authChanges.value
                scope.launch {
                    try {
                        val message = try {
                            val response = RetrofitClient.apiService.getPhotographerPhotoDownload("Bearer $token", photo.id)
                            val url = response.data?.url?.takeIf { response.success }
                                ?: error(response.error ?: "Download unavailable.")
                            if (generation != session.authChanges.value || viewModel.previewState.value.photoId != photo.id) return@launch
                            when (val saved = PhotoDownloader.saveToGallery(context, url, PhotoDownloader.buildFilename(photo.id, state.photo?.bib))) {
                                is PhotoDownloader.Result.Saved -> "Saved ${saved.displayName} to your gallery."
                                is PhotoDownloader.Result.Failed -> saved.message
                            }
                        } catch (e: Exception) {
                            if (generation == session.authChanges.value && viewModel.previewState.value.photoId == photo.id && downloadingPhotoId == photo.id)
                                downloadErrorPhotoId = photo.id
                            RetrofitClient.parseError(e)
                        }
                        if (generation == session.authChanges.value && viewModel.previewState.value.photoId == photo.id && downloadingPhotoId == photo.id)
                            Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                    } finally {
                        if (downloadingPhotoId == photo.id) downloadingPhotoId = null
                    }
                }
            }
        },
        details = {
            PurchaseDetails(
                state = state,
                transaction = transaction?.takeIf { it.photoId == photoId },
                onOpenEvent = { id -> close(); onOpenEvent(id) },
                onSelectPurchase = viewModel::selectPurchase,
                onRetry = viewModel::retryPreview,
                onLoadMore = viewModel::loadMorePurchases,
                downloadError = downloadErrorPhotoId == photoId,
            )
        },
    )
}

@Composable
private fun PurchaseDetails(
    state: PurchasePreviewState,
    transaction: PhotographerTransactionDto?,
    onOpenEvent: (String) -> Unit,
    onSelectPurchase: (String) -> Unit,
    onRetry: () -> Unit,
    onLoadMore: () -> Unit,
    downloadError: Boolean,
) {
    val selected = state.selectedPurchase
    val eventId = state.event?.id ?: transaction?.eventId
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Kicker("Purchase details", color = Ink)
        Fact("Buyer", selected?.buyer ?: transaction?.buyer ?: "Unavailable")
        Fact("Reference", (selected?.orderId ?: state.orderId ?: transaction?.orderId)?.let(::formatOrderRef) ?: "Unavailable")
        Fact("Purchased", formatManila(selected?.purchasedAt ?: transaction?.paidAt, "MMM d, yyyy · h:mm a") ?: "Unavailable")
        if (selected != null && selected.purchasedAt == null) Fact("Order created", formatManila(selected.orderCreatedAt, "MMM d, yyyy · h:mm a") ?: "Unavailable")
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Event", style = Typography.bodyMedium, color = Slate)
            Text(
                state.event?.name ?: transaction?.eventName ?: "Event archived",
                style = Typography.bodyMedium,
                color = Ink,
                modifier = Modifier.then(if (eventId != null) Modifier.clickable { onOpenEvent(eventId) }.heightIn(min = 48.dp) else Modifier),
            )
        }
        Fact("Photo", state.photo?.bib?.let { "Bib $it" } ?: state.photoId.orEmpty())
        Fact("Photo price", selected?.pricePhp?.let(::peso) ?: "Unavailable")
        Fact("Discount", selected?.discountPhp?.let(::peso) ?: "Unavailable")
        Fact("Paid for this photo", selected?.let { if (it.kind == "free") "Free (${peso(it.paidPhp)})" else peso(it.paidPhp) } ?: "Unavailable")
        Fact("Refunded", selected?.refundedPhp?.let(::peso) ?: "Unavailable")
        Fact("Refund status", selected?.refundStatus?.replaceFirstChar { it.uppercase() } ?: "Unavailable")
        selected?.refundedAt?.let { Fact("Refund completed", formatManila(it, "MMM d, yyyy · h:mm a") ?: "Unavailable") }
            ?: selected?.transactions?.lastOrNull { it.isRefund }?.recordedAt?.let { Fact("Refund recorded", formatManila(it, "MMM d, yyyy · h:mm a") ?: "Unavailable") }
        Fact("Your net earnings", selected?.netKeptPhp?.let(::peso) ?: "Unavailable")
        if (selected?.refundPending == true) Text("Refund pending", style = Typography.bodyMedium, color = Slate)
        selected?.transactions?.forEach { row ->
            Text(
                "${if (row.id == state.transactionId) "Selected · " else ""}${if (row.isRefund) "Earnings adjustment" else "Sale earnings"} · ${signedPeso(row.amountKept)} · Recorded ${formatManila(row.recordedAt, "MMM d, yyyy · h:mm a") ?: row.recordedAt}",
                style = Typography.bodyMedium,
                color = Ink,
            )
        }
        if (state.isLoading) Text("Loading purchases…", style = Typography.bodyMedium, color = Slate)
        if (state.error != null || (!state.isLoading && selected == null)) {
            Text(if (state.errorStatus == 404) "Purchase unavailable." else state.error ?: "Purchase unavailable.", style = Typography.bodyMedium, color = ErrorRed)
            GhostCta(text = "Retry purchases", onClick = onRetry, modifier = Modifier.fillMaxWidth())
        }
        if (state.items.size > 1 || state.total > 1) {
            Kicker("Purchase history", color = Ink)
            (listOfNotNull(selected) + state.items).distinctBy { it.orderId to it.photoId }.forEach { purchase ->
                Text(
                    "${purchase.buyer} · ${formatOrderRef(purchase.orderId)}",
                    style = Typography.bodyMedium,
                    color = Ink,
                    modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp).clickable { onSelectPurchase(purchase.orderId) }.padding(vertical = 12.dp),
                )
            }
        }
        state.appendError?.let { Text("Couldn't load more purchases.", style = Typography.bodyMedium, color = ErrorRed) }
        if (state.hasMore) GhostCta(
            text = if (state.appendError != null) "Retry load more" else if (state.isLoadingMore) "Loading…" else "Load more purchases",
            enabled = !state.isLoadingMore,
            onClick = onLoadMore,
            modifier = Modifier.fillMaxWidth(),
        )
        if (downloadError) Text("Download unavailable. Try again.", style = Typography.bodyMedium, color = ErrorRed)
    }
}

@Composable
private fun Fact(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = Typography.bodyMedium, color = Slate)
        Text(value, style = Typography.bodyMedium, color = Ink, modifier = Modifier.padding(start = 12.dp))
    }
}

private fun signedPeso(amount: Double) = "${if (amount < 0) "-" else "+"}${peso(abs(amount))}"

@Preview(showBackground = true)
@Composable
private fun PurchaseDetailsUnavailablePreview() {
    StudioTheme {
        PurchaseDetails(
            state = PurchasePreviewState(photoId = "photo-a"),
            transaction = null,
            onOpenEvent = {},
            onSelectPurchase = {},
            onRetry = {},
            onLoadMore = {},
            downloadError = false,
        )
    }
}
