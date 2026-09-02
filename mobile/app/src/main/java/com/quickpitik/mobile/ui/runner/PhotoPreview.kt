package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.KeyboardArrowLeft
import androidx.compose.material.icons.filled.KeyboardArrowRight
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.CartItemDto
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.ArrowLabel
import com.quickpitik.mobile.ui.theme.BadgeShape
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BrandLogo
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.NumeralStyle
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.TileShape
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.Typography

// Unified photo lightbox — the one preview surface used by both the cart sheet
// and the gallery photo-details flow. Mirrors website PhotoPreviewCard's
// "browse" mode where `isInCart(photo)` flips the button labels between
// Add+Buy and Remove+Checkout per-photo.
//
// The image area is a HorizontalPager so the runner can swipe between photos;
// header/stats/price/buttons all reflect `photos[currentIndex]`. Prev/Next nav
// row + the index counter are hidden when there is only one photo.

data class PhotoPreviewData(
    val id: String,
    val price: Double,
    val imageUrl: String?,
    val eventName: String?,
    val salesCount: Int? = null,
    // Photographer credit. Null name = a legacy/seed photo with no
    // photographerId, so no byline renders at all. Non-null name with a null
    // handle = an unverified photographer (the handle is assigned at
    // verification) — the byline renders as plain text, never a tap target.
    val photographerHandle: String? = null,
    val photographerName: String? = null,
    // LocalReview mode only — one line under the frame (filename · sync state).
    val caption: String? = null,
)

// PhotoPreview has two flavors. Browse is the runner buy-flow (Add to cart /
// Buy now). OwnerReview is for the photographer reviewing their own uploads
// from the event-share page — same Dialog/Pager shell, no purchase UI,
// "X sold" stat in place of price.
// Owned is the runner's post-purchase view (website PhotoPreviewCard
// mode="owned"): no price, no cart — just "Yours to keep" and a download.
// LocalReview: the photographer inspecting a frame still on the phone (the
// Capture tab's sync strip) — imageUrl is a file:// URI, no server, no commerce.
enum class PhotoPreviewMode { Browse, OwnerReview, Owned, LocalReview }

fun PhotoDto.toPreviewData(eventName: String?): PhotoPreviewData = PhotoPreviewData(
    id = id,
    price = price,
    // cleanUrl is non-null only on photos the runner owns, so this is the
    // owned-mode swap the website does with `cleanUrl ?? imageUrl`.
    imageUrl = cleanUrl ?: imageUrl,
    eventName = eventName,
    photographerHandle = photographerHandle,
    photographerName = photographerName,
)

fun CartItemDto.toPreviewData(): PhotoPreviewData = PhotoPreviewData(
    id = photoId,
    price = price,
    imageUrl = thumbnailUrl,
    eventName = eventName,
)

/**
 * "Photo by {name}" credit in the lightbox. Tappable only when the
 * photographer has a handle — an unverified photographer has a name and no
 * handle, and linking one would route to /{null}. Carries no Fresh: the
 * Browse viewport already spends its one accent on the Buy CTA.
 */
@Composable
private fun PhotographerByline(
    name: String,
    handle: String?,
    onOpen: (String) -> Unit,
) {
    Column(
        modifier = Modifier
            .clip(BadgeShape)
            .then(
                if (handle != null) Modifier.clickable { onOpen(handle) } else Modifier,
            )
            // Padding inside the clickable so the tap target clears 48dp:
            // kicker (~16dp) + name (~20dp) + 16dp padding.
            .padding(horizontal = 8.dp, vertical = 8.dp)
            .widthIn(max = 180.dp),
        horizontalAlignment = Alignment.End,
    ) {
        Kicker("Photo by", color = SlateSoft)
        Spacer(modifier = Modifier.height(2.dp))
        if (handle != null) {
            ArrowLabel(
                text = "$name →",
                color = Ink,
                style = Typography.bodyMedium,
                iconSize = 12.dp,
            )
        } else {
            Text(
                text = name,
                style = Typography.bodyMedium,
                color = Ink,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                textAlign = TextAlign.End,
            )
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun PhotoPreview(
    photos: List<PhotoPreviewData>,
    currentIndex: Int,
    onClose: () -> Unit,
    onIndexChange: (Int) -> Unit,
    isInCart: (PhotoPreviewData) -> Boolean = { false },
    onToggleCart: (PhotoPreviewData) -> Unit = {},
    onBuyNow: (PhotoPreviewData) -> Unit = {},
    // Owned + OwnerReview modes — saves the un-watermarked original to the
    // gallery. Owned resolves it from the runner's purchase grant; OwnerReview
    // from the photographer's own /photos/{id}/download endpoint.
    onDownload: (PhotoPreviewData) -> Unit = {},
    // Browse mode only — opens the photographer's public profile. Called with a
    // non-null handle; the byline is not a tap target without one.
    onOpenPhotographer: (String) -> Unit = {},
    mode: PhotoPreviewMode = PhotoPreviewMode.Browse,
    // Browse mode only. False for a PHOTOGRAPHER browsing in runner view
    // (ViewMode): the cart endpoints are RUNNER-role-gated server-side, so the
    // Add/Buy CTAs are hidden — price + byline stay.
    commerceEnabled: Boolean = true,
) {
    if (photos.isEmpty()) {
        onClose()
        return
    }
    val safeIndex = currentIndex.coerceIn(0, photos.size - 1)
    val activePhoto = photos[safeIndex]
    val photoInCart = isInCart(activePhoto)
    val hasMultiple = photos.size > 1

    val pagerState = rememberPagerState(
        initialPage = safeIndex,
        pageCount = { photos.size },
    )

    // Pager → parent: settled-page emissions update the parent's currentIndex.
    // rememberUpdatedState lets the long-running collector see the latest
    // currentIndex/onIndexChange without restarting (which would skip events).
    val latestOnIndexChange by rememberUpdatedState(onIndexChange)
    val latestCurrentIndex by rememberUpdatedState(currentIndex)
    LaunchedEffect(pagerState) {
        snapshotFlow { pagerState.currentPage }.collect { page ->
            if (page != latestCurrentIndex) latestOnIndexChange(page)
        }
    }
    // Parent → pager: prev/next button taps push currentIndex; animate the
    // pager to catch up. Guarded so swipe-driven changes don't re-animate.
    LaunchedEffect(currentIndex) {
        if (currentIndex in photos.indices &&
            pagerState.currentPage != currentIndex
        ) {
            pagerState.animateScrollToPage(currentIndex)
        }
    }

    val scrimInteraction = remember { MutableInteractionSource() }
    val cardInteraction = remember { MutableInteractionSource() }

    Dialog(
        onDismissRequest = onClose,
        properties = DialogProperties(
            usePlatformDefaultWidth = false,
            dismissOnBackPress = true,
            dismissOnClickOutside = true,
        ),
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Ink.copy(alpha = 0.92f))
                .clickable(
                    onClick = onClose,
                    indication = null,
                    interactionSource = scrimInteraction,
                ),
        ) {
            Box(
                modifier = Modifier
                    .align(Alignment.Center)
                    .fillMaxWidth(0.94f)
                    .fillMaxHeight(0.92f)
                    .clip(QpCardShape)
                    .background(Bone)
                    .clickable(
                        onClick = {},
                        indication = null,
                        interactionSource = cardInteraction,
                    ),
            ) {
                Column(modifier = Modifier.fillMaxSize()) {
                    // ── Header ──────────────────────────────────────
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Kicker(
                            text = activePhoto.eventName ?: "QuickPitik",
                            modifier = Modifier.weight(1f),
                            color = Ink,
                        )
                        if (hasMultiple) {
                            Text(
                                text = "${safeIndex + 1} / ${photos.size}",
                                style = NumeralStyle.copy(fontSize = 12.sp),
                                color = SlateSoft,
                            )
                            Spacer(modifier = Modifier.width(12.dp))
                        }
                        Box(
                            modifier = Modifier
                                .size(36.dp)
                                .clip(CircleShape)
                                .border(1.dp, Line, CircleShape)
                                .clickable(onClick = onClose),
                            contentAlignment = Alignment.Center,
                        ) {
                            Icon(
                                imageVector = Icons.Default.Close,
                                contentDescription = "Close preview",
                                tint = Ink,
                                modifier = Modifier.size(14.dp),
                            )
                        }
                    }
                    Divider(color = Line)

                    // ── Pager image area ────────────────────────────
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth()
                            .background(Ink),
                    ) {
                        HorizontalPager(
                            state = pagerState,
                            modifier = Modifier.fillMaxSize(),
                        ) { page ->
                            val pagePhoto = photos[page]
                            Box(modifier = Modifier.fillMaxSize()) {
                                if (pagePhoto.imageUrl != null) {
                                    AsyncImage(
                                        model = RetrofitClient.resolveImageUrl(pagePhoto.imageUrl),
                                        contentDescription = "Race photo",
                                        contentScale = ContentScale.Fit,
                                        modifier = Modifier.fillMaxSize(),
                                    )
                                } else {
                                    Box(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .background(InkSoft),
                                        contentAlignment = Alignment.Center,
                                    ) {
                                        Text(
                                            text = "QUICKPITIK PREVIEW",
                                            style = Typography.labelMedium,
                                            color = Color.White.copy(alpha = 0.35f),
                                        )
                                    }
                                }
                                // QuickPitik watermark (bottom-left of each
                                // page) — Browse only. A runner in Owned mode
                                // PAID for the unwatermarked shot, and a
                                // photographer reviewing their own uploads
                                // needs no client-side mark either (web
                                // suppresses it when cleanUrl is present).
                                if (mode == PhotoPreviewMode.Browse) {
                                    BrandLogo(
                                        compact = true,
                                        contentDescription = null,
                                        modifier = Modifier
                                            .align(Alignment.BottomStart)
                                            .padding(12.dp)
                                            .alpha(0.72f)
                                            .clip(TileShape),
                                    )
                                }
                            }
                        }
                        // "In cart" pill — overlay outside the pager so it doesn't
                        // scroll with the photo. Reflects the active page's status.
                        // Browse-mode only; the OwnerReview flow has no cart.
                        if (photoInCart && mode == PhotoPreviewMode.Browse) {
                            Row(
                                modifier = Modifier
                                    .align(Alignment.TopEnd)
                                    .padding(12.dp)
                                    .clip(PillShape)
                                    .background(Fresh)
                                    .padding(horizontal = 12.dp, vertical = 6.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(6.dp)
                                        .clip(CircleShape)
                                        .background(Color.White),
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = "IN CART",
                                    style = Typography.labelSmall,
                                    color = Color.White,
                                )
                            }
                        }
                    }

                    // ── Bottom row ──────────────────────────────────
                    // Browse: price + Add/Buy actions (runner buy-flow).
                    // OwnerReview: "X sold" stat + single Close (photographer
                    // reviewing their own uploads — no purchase UI).
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp, vertical = 16.dp),
                    ) {
                        when (mode) {
                            PhotoPreviewMode.Browse -> {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.Bottom,
                                ) {
                                    Column(modifier = Modifier.weight(1f)) {
                                        Kicker("Price", color = SlateSoft)
                                        Spacer(modifier = Modifier.height(2.dp))
                                        Text(
                                            text = "₱${"%,.2f".format(activePhoto.price)}",
                                            style = NumeralStyle.copy(fontSize = 22.sp),
                                            color = Ink,
                                        )
                                    }
                                    // Credit sits in the dead half of the price
                                    // row, so it costs the image no height.
                                    if (activePhoto.photographerName != null) {
                                        PhotographerByline(
                                            name = activePhoto.photographerName,
                                            handle = activePhoto.photographerHandle,
                                            onOpen = onOpenPhotographer,
                                        )
                                    }
                                }
                                if (commerceEnabled) {
                                    Spacer(modifier = Modifier.height(12.dp))
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                                    ) {
                                        GhostCta(
                                            text = if (photoInCart) "Remove" else "Add to cart",
                                            onClick = { onToggleCart(activePhoto) },
                                            modifier = Modifier.weight(1f),
                                        )
                                        PrimaryCta(
                                            text = if (photoInCart) "Checkout →" else "Buy now →",
                                            onClick = { onBuyNow(activePhoto) },
                                            modifier = Modifier.weight(1f),
                                        )
                                    }
                                }
                            }
                            PhotoPreviewMode.Owned -> {
                                Kicker(
                                    text = "Yours to keep",
                                    color = SlateSoft,
                                    modifier = Modifier.align(Alignment.CenterHorizontally),
                                )
                                Spacer(modifier = Modifier.height(12.dp))
                                PrimaryCta(
                                    text = "Download photo ↓",
                                    onClick = { onDownload(activePhoto) },
                                    modifier = Modifier.fillMaxWidth(),
                                )
                            }
                            PhotoPreviewMode.LocalReview -> {
                                activePhoto.caption?.let { caption ->
                                    Text(
                                        text = caption,
                                        style = Typography.bodyMedium,
                                        color = Slate,
                                    )
                                    Spacer(modifier = Modifier.height(12.dp))
                                }
                                GhostCta(
                                    text = "Close",
                                    onClick = onClose,
                                    modifier = Modifier.fillMaxWidth(),
                                )
                            }
                            PhotoPreviewMode.OwnerReview -> {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.Bottom,
                                ) {
                                    Column(modifier = Modifier.weight(1f)) {
                                        Kicker("Sold", color = SlateSoft)
                                        Spacer(modifier = Modifier.height(2.dp))
                                        Text(
                                            text = (activePhoto.salesCount ?: 0).toString(),
                                            style = NumeralStyle.copy(fontSize = 22.sp),
                                            color = Ink,
                                        )
                                    }
                                }
                                Spacer(modifier = Modifier.height(12.dp))
                                // Same two-CTA row as Browse mode. The download
                                // resolves a presigned URL for the photographer's
                                // own un-watermarked original — website parity with
                                // /dashboard/events/[id]'s per-photo download.
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                                ) {
                                    GhostCta(
                                        text = "Close",
                                        onClick = onClose,
                                        modifier = Modifier.weight(1f),
                                    )
                                    PrimaryCta(
                                        text = "Download ↓",
                                        onClick = { onDownload(activePhoto) },
                                        modifier = Modifier.weight(1f),
                                    )
                                }
                            }
                        }
                    }

                    // ── Prev / index / Next nav (multi-photo only) ──
                    if (hasMultiple) {
                        Divider(color = Line)
                        val canPrev = safeIndex > 0
                        val canNext = safeIndex < photos.size - 1
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 20.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Row(
                                modifier = Modifier
                                    .weight(1f)
                                    .clickable(enabled = canPrev) {
                                        onIndexChange(safeIndex - 1)
                                    }
                                    .padding(vertical = 8.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Icon(
                                    imageVector = Icons.Default.KeyboardArrowLeft,
                                    contentDescription = "Previous",
                                    tint = if (canPrev) Ink else SlateSoft,
                                    modifier = Modifier.size(20.dp),
                                )
                                Text(
                                    text = "Prev",
                                    style = Typography.labelMedium,
                                    color = if (canPrev) Ink else SlateSoft,
                                )
                            }
                            Text(
                                text = "${safeIndex + 1} / ${photos.size}",
                                style = NumeralStyle.copy(fontSize = 12.sp),
                                color = SlateSoft,
                                textAlign = TextAlign.Center,
                            )
                            Row(
                                modifier = Modifier
                                    .weight(1f)
                                    .clickable(enabled = canNext) {
                                        onIndexChange(safeIndex + 1)
                                    }
                                    .padding(vertical = 8.dp),
                                horizontalArrangement = Arrangement.End,
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Text(
                                    text = "Next",
                                    style = Typography.labelMedium,
                                    color = if (canNext) Ink else SlateSoft,
                                )
                                Icon(
                                    imageVector = Icons.Default.KeyboardArrowRight,
                                    contentDescription = "Next",
                                    tint = if (canNext) Ink else SlateSoft,
                                    modifier = Modifier.size(20.dp),
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
