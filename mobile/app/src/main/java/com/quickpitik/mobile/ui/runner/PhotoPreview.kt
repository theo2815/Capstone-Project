package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.CartItemDto
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.ui.theme.*

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
)

// PhotoPreview has two flavors. Browse is the runner buy-flow (Add to cart /
// Buy now). OwnerReview is for the photographer reviewing their own uploads
// from the event-share page — same Dialog/Pager shell, no purchase UI,
// "X sold" stat in place of price.
enum class PhotoPreviewMode { Browse, OwnerReview }

fun PhotoDto.toPreviewData(eventName: String?): PhotoPreviewData = PhotoPreviewData(
    id = id,
    price = price,
    imageUrl = imageUrl,
    eventName = eventName,
)

fun CartItemDto.toPreviewData(): PhotoPreviewData = PhotoPreviewData(
    id = photoId,
    price = price,
    imageUrl = thumbnailUrl,
    eventName = eventName,
)

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
    mode: PhotoPreviewMode = PhotoPreviewMode.Browse,
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
                                        model = pagePhoto.imageUrl,
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
                                // QuickPitik watermark (bottom-left of each page).
                                Row(
                                    modifier = Modifier
                                        .align(Alignment.BottomStart)
                                        .padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Box(
                                        modifier = Modifier
                                            .size(12.dp)
                                            .background(Color.White.copy(alpha = 0.35f)),
                                    )
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(
                                        text = "QUICKPITIK",
                                        style = Typography.labelSmall,
                                        color = Color.White.copy(alpha = 0.55f),
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
                                }
                                Spacer(modifier = Modifier.height(12.dp))
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                                ) {
                                    GhostCta(
                                        text = if (photoInCart) "− Remove" else "+ Add to cart",
                                        onClick = { onToggleCart(activePhoto) },
                                        modifier = Modifier.weight(1f),
                                    )
                                    PrimaryCta(
                                        text = if (photoInCart) "Checkout now →" else "Buy now →",
                                        onClick = { onBuyNow(activePhoto) },
                                        modifier = Modifier.weight(1f),
                                    )
                                }
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
                                GhostCta(
                                    text = "Close",
                                    onClick = onClose,
                                    modifier = Modifier.fillMaxWidth(),
                                )
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
