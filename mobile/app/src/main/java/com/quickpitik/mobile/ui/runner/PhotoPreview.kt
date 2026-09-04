package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.KeyboardArrowLeft
import androidx.compose.material.icons.filled.KeyboardArrowRight
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import coil.compose.AsyncImage
import coil.compose.AsyncImagePainter
import coil.request.ImageRequest
import coil.size.Size
import kotlinx.coroutines.launch
import com.quickpitik.mobile.data.remote.CartItemDto
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.ArrowLabel
import com.quickpitik.mobile.ui.theme.BadgeShape
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.SecureScreen
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.NavChrome
import com.quickpitik.mobile.ui.theme.NumeralStyle
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.Typography

// Unified photo lightbox — the one preview surface used by both the cart sheet
// and the gallery photo-details flow. Mirrors website PhotoPreviewCard's
// below-`lg` layout: a photograph on an ink stage with the chrome floating
// over it (counter top-left, close top-right, prev/next at the vertical centre
// where a thumb can reach) and one bone strip underneath carrying event ·
// price · credit · CTAs · hint. A portrait photo sizes the stage to its own
// aspect so it fills edge to edge; a landscape photo keeps the full-height
// stage and its letterbox.
//
// The image area is a HorizontalPager so the runner can swipe between photos;
// strip and chevrons all reflect `photos[currentIndex]`.

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
 * "Photo by @handle →" credit line in the strip. Tappable only when the
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
    Row(
        modifier = Modifier
            .clip(BadgeShape)
            .then(
                if (handle != null) Modifier.clickable { onOpen(handle) } else Modifier,
            )
            // One kicker line is ~16dp; the min height keeps the tap target at 48dp.
            .heightIn(min = 48.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Kicker("Photo by", color = SlateSoft)
        Spacer(modifier = Modifier.width(6.dp))
        if (handle != null) {
            ArrowLabel(
                text = "@${handle.uppercase()} →",
                color = Ink,
                style = Typography.labelMedium,
                iconSize = 12.dp,
            )
        } else {
            Kicker(name, color = Ink)
        }
    }
}

/** 48dp white pill button floating over the photo (close, prev, next). */
@Composable
private fun ChromeButton(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .size(48.dp)
            .clip(CircleShape)
            .background(NavChrome)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            imageVector = icon,
            contentDescription = contentDescription,
            tint = Ink,
            modifier = Modifier.size(20.dp),
        )
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
    val canPrev = safeIndex > 0
    val canNext = safeIndex < photos.size - 1

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
    // Parent → pager: chevron taps push currentIndex; animate the pager to
    // catch up. Guarded so swipe-driven changes don't re-animate.
    LaunchedEffect(currentIndex) {
        if (currentIndex in photos.indices &&
            pagerState.currentPage != currentIndex
        ) {
            pagerState.animateScrollToPage(currentIndex)
        }
    }

    // Width / height of each served image, reported by the page once Coil has
    // it. The active page's ratio decides whether the stage fits the photo.
    val aspects = remember { mutableStateMapOf<Int, Float>() }
    val aspect = aspects[safeIndex]
    val fitPortrait = aspect != null && aspect < 1f

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
        // The Dialog is its own Window: Browse = unpurchased preview, so block
        // screenshots of it here (the host screen's flag doesn't reach it).
        if (mode == PhotoPreviewMode.Browse) SecureScreen()
        BoxWithConstraints(
            modifier = Modifier
                .fillMaxSize()
                .background(Ink.copy(alpha = 0.92f))
                .clickable(
                    onClick = onClose,
                    indication = null,
                    interactionSource = scrimInteraction,
                ),
            contentAlignment = Alignment.Center,
        ) {
            val cardWidth = maxWidth * CARD_WIDTH_FRACTION
            val cardMaxHeight = maxHeight * CARD_HEIGHT_FRACTION
            Column(
                modifier = Modifier
                    .width(cardWidth)
                    // Portrait: the card wraps the fitted stage + strip and only
                    // caps at the max. Landscape / not yet loaded: full height,
                    // the stage takes what the strip leaves.
                    .then(
                        if (fitPortrait) Modifier.heightIn(max = cardMaxHeight)
                        else Modifier.height(cardMaxHeight),
                    )
                    .clip(QpCardShape)
                    .background(Bone)
                    .clickable(
                        onClick = {},
                        indication = null,
                        interactionSource = cardInteraction,
                    ),
            ) {
                // ── Stage ───────────────────────────────────────────
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f, fill = !fitPortrait)
                        .then(
                            if (fitPortrait) Modifier.height(cardWidth / aspect!!) else Modifier,
                        )
                        .background(Ink),
                ) {
                    HorizontalPager(
                        state = pagerState,
                        modifier = Modifier.fillMaxSize(),
                    ) { page ->
                        val pagePhoto = photos[page]
                        Box(modifier = Modifier.fillMaxSize()) {
                            if (pagePhoto.imageUrl != null) {
                                ZoomableImage(
                                    url = RetrofitClient.resolveImageUrl(pagePhoto.imageUrl)!!,
                                    onAspect = { aspects[page] = it },
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
                            // No client-side platform mark: the backend
                            // bakes the QuickPitik credit + photographer
                            // logo into imageUrl; cleanUrl (owned) is
                            // deliberately unmarked.
                        }
                    }

                    // Chrome floats outside the pager so it doesn't scroll
                    // with the photo.
                    if (hasMultiple) {
                        Box(
                            modifier = Modifier
                                .align(Alignment.TopStart)
                                .padding(12.dp)
                                .height(48.dp)
                                .clip(PillShape)
                                .background(NavChrome)
                                .padding(horizontal = 16.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(
                                text = "${safeIndex + 1} / ${photos.size}",
                                style = NumeralStyle.copy(fontSize = 13.sp),
                                color = Ink,
                            )
                        }
                    }
                    ChromeButton(
                        icon = Icons.Default.Close,
                        contentDescription = "Close preview",
                        onClick = onClose,
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(12.dp),
                    )
                    if (canPrev) {
                        ChromeButton(
                            icon = Icons.Default.KeyboardArrowLeft,
                            contentDescription = "Previous",
                            onClick = { onIndexChange(safeIndex - 1) },
                            modifier = Modifier
                                .align(Alignment.CenterStart)
                                .padding(12.dp),
                        )
                    }
                    if (canNext) {
                        ChromeButton(
                            icon = Icons.Default.KeyboardArrowRight,
                            contentDescription = "Next",
                            onClick = { onIndexChange(safeIndex + 1) },
                            modifier = Modifier
                                .align(Alignment.CenterEnd)
                                .padding(12.dp),
                        )
                    }
                }

                // ── Strip ───────────────────────────────────────────
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 20.dp, vertical = 16.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.Top,
                    ) {
                        Kicker(
                            text = activePhoto.eventName ?: "QuickPitik",
                            modifier = Modifier.weight(1f),
                            color = Ink,
                        )
                        Spacer(modifier = Modifier.width(16.dp))
                        when (mode) {
                            PhotoPreviewMode.Browse -> Text(
                                text = "₱${"%,.2f".format(activePhoto.price)}",
                                style = NumeralStyle.copy(fontSize = 22.sp),
                                color = Ink,
                            )
                            PhotoPreviewMode.Owned -> Kicker("Yours to keep", color = SlateSoft)
                            PhotoPreviewMode.OwnerReview -> Column(horizontalAlignment = Alignment.End) {
                                Text(
                                    text = (activePhoto.salesCount ?: 0).toString(),
                                    style = NumeralStyle.copy(fontSize = 22.sp),
                                    color = Ink,
                                )
                                Kicker("Sold", color = SlateSoft)
                            }
                            PhotoPreviewMode.LocalReview -> Unit
                        }
                    }
                    if (mode == PhotoPreviewMode.Browse && activePhoto.photographerName != null) {
                        PhotographerByline(
                            name = activePhoto.photographerName,
                            handle = activePhoto.photographerHandle,
                            onOpen = onOpenPhotographer,
                        )
                    }

                    when (mode) {
                        PhotoPreviewMode.Browse -> if (commerceEnabled) {
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
                            if (photoInCart) {
                                Spacer(modifier = Modifier.height(12.dp))
                                Kicker(
                                    text = "✓ In cart",
                                    color = Fresh,
                                    modifier = Modifier.align(Alignment.CenterHorizontally),
                                )
                            }
                        }
                        PhotoPreviewMode.Owned -> {
                            Spacer(modifier = Modifier.height(12.dp))
                            PrimaryCta(
                                text = "Download photo ↓",
                                onClick = { onDownload(activePhoto) },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                        PhotoPreviewMode.LocalReview -> {
                            activePhoto.caption?.let { caption ->
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    text = caption,
                                    style = Typography.bodyMedium,
                                    color = Slate,
                                )
                            }
                            Spacer(modifier = Modifier.height(12.dp))
                            GhostCta(
                                text = "Close",
                                onClick = onClose,
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                        PhotoPreviewMode.OwnerReview -> {
                            Spacer(modifier = Modifier.height(12.dp))
                            // The download resolves a presigned URL for the
                            // photographer's own un-watermarked original —
                            // website parity with /dashboard/events/[id].
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

                    // Gesture hint — invisible gestures don't ship.
                    if (activePhoto.imageUrl != null) {
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(
                            text = "PINCH OR DOUBLE-TAP TO ZOOM",
                            style = Typography.labelMedium,
                            color = SlateSoft,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                }
            }
        }
    }
}

// Pinch-to-zoom + double-tap-to-zoom over the preview. Pans while zoomed and
// hands the drag back to the HorizontalPager at 1×, so a swipe still pages.
// Decodes at the preview's native size (Size.ORIGINAL) — Coil would otherwise
// downsample to the view and a 3× zoom would be a blur of the blur.
//
// What is zoomed is exactly what was served: the watermarked preview for a
// browser, the clean original only for an owner (cleanUrl). Zoom never
// requests a different asset.
@Composable
private fun ZoomableImage(
    url: String,
    onAspect: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    val scope = rememberCoroutineScope()
    val scale = remember { Animatable(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    var size by remember { mutableStateOf(IntSize.Zero) }

    fun clamp(o: Offset, s: Float): Offset {
        val maxX = (size.width * (s - 1)) / 2
        val maxY = (size.height * (s - 1)) / 2
        return Offset(o.x.coerceIn(-maxX, maxX), o.y.coerceIn(-maxY, maxY))
    }

    Box(
        modifier = modifier
            .onSizeChanged { size = it }
            .pointerInput(Unit) {
                detectTapGestures(
                    onDoubleTap = { tap ->
                        val target = if (scale.value > 1.05f) 1f else DOUBLE_TAP_SCALE
                        val centre = Offset(size.width / 2f, size.height / 2f)
                        // Zoom into the tapped point; the offset moves so that point stays put.
                        val next = if (target == 1f) Offset.Zero else clamp((centre - tap) * (target - 1), target)
                        scope.launch {
                            offset = next
                            scale.animateTo(target, tween(ZOOM_ANIM_MS))
                        }
                    },
                )
            }
            .pointerInput(Unit) {
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    do {
                        val event = awaitPointerEvent()
                        val zoom = event.calculateZoom()
                        val pan = event.calculatePan()
                        val pinching = event.changes.size > 1
                        if (pinching || scale.value > 1f) {
                            val s = (scale.value * zoom).coerceIn(1f, MAX_SCALE)
                            val centroid = event.calculateCentroid()
                            val centre = Offset(size.width / 2f, size.height / 2f)
                            // Keep the pinch centroid fixed while scaling around the centre.
                            val scaled = (offset - (centroid - centre)) * (s / scale.value) + (centroid - centre)
                            offset = clamp(scaled + pan, s)
                            scope.launch { scale.snapTo(s) }
                            event.changes.forEach { if (it.positionChanged()) it.consume() }
                        }
                    } while (event.changes.any { it.pressed })
                    if (scale.value <= 1.02f) {
                        scope.launch { offset = Offset.Zero; scale.snapTo(1f) }
                    }
                }
            },
    ) {
        AsyncImage(
            model = ImageRequest.Builder(LocalContext.current).data(url).size(Size.ORIGINAL).build(),
            contentDescription = "Race photo",
            contentScale = ContentScale.Fit,
            onSuccess = { state: AsyncImagePainter.State.Success ->
                val d = state.result.drawable
                if (d.intrinsicWidth > 0 && d.intrinsicHeight > 0) {
                    onAspect(d.intrinsicWidth.toFloat() / d.intrinsicHeight)
                }
            },
            modifier = Modifier
                .fillMaxSize()
                .graphicsLayer {
                    scaleX = scale.value
                    scaleY = scale.value
                    translationX = offset.x
                    translationY = offset.y
                },
        )
    }
}

private const val CARD_WIDTH_FRACTION = 0.94f
private const val CARD_HEIGHT_FRACTION = 0.92f
private const val DOUBLE_TAP_SCALE = 2.5f
private const val MAX_SCALE = 4f
private const val ZOOM_ANIM_MS = 220
