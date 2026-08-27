package com.quickpitik.mobile.ui.photographer

import android.graphics.Color as AndroidColor
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.CoverSourceDto
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.PhotographerEventCoverageDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import androidx.compose.runtime.saveable.rememberSaveable
import com.quickpitik.mobile.ui.runner.CartViewModel
import com.quickpitik.mobile.ui.runner.PhotoPreview
import com.quickpitik.mobile.ui.runner.rememberIsTrueRunner
import com.quickpitik.mobile.ui.runner.toPreviewData
import com.quickpitik.mobile.ui.theme.*

// Mobile mirror of website /{handle} (public photographer profile) and
// /{handle}/events/[slug] (per-event public gallery). Two entry points:
//   • the photographer's own "Preview public profile" (Overview tab), which
//     passes the handle off their brand settings — null until verification;
//   • a runner tapping the photographer byline in the photo lightbox, which
//     routes here with the handle off PhotoDto (2026-08-15).
// [handle] is therefore a parameter rather than something read off the
// logged-in photographer — a runner has no brand settings to read.
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotographerPublicProfileScreen(
    handle: String?,
    viewModel: PublicPhotographerViewModel,
    onBack: () -> Unit,
    // Present only in RUNNER contexts (the photographer/{handle} route) —
    // enables the gallery's add-to-cart flow. The studio's own profile
    // preview passes nothing and stays a viewer.
    cartViewModel: CartViewModel? = null,
) {
    val profileState by viewModel.publicProfileState.collectAsState()
    val resolvedHandle = handle?.takeIf { it.isNotBlank() }
    var selectedEventSlug by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(resolvedHandle) {
        if (resolvedHandle != null) viewModel.fetchPublicProfile(resolvedHandle)
    }

    val gallerySlug = selectedEventSlug
    if (resolvedHandle != null && gallerySlug != null) {
        ProfileEventGalleryView(
            viewModel = viewModel,
            handle = resolvedHandle,
            slug = gallerySlug,
            onBack = { selectedEventSlug = null },
            cartViewModel = cartViewModel,
        )
        return
    }

    Surface(modifier = Modifier.fillMaxSize(), color = Bone) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(20.dp)
        ) {
            BackRow(label = "BACK", onBack = onBack)
            Spacer(modifier = Modifier.height(16.dp))

            // Only reachable from the photographer's own preview: the runner
            // byline is a tap target only when the handle is non-null, so a
            // runner can never land here handle-less.
            if (resolvedHandle == null) {
                EmptyStateCard("Set your public handle in the Settings tab to preview your profile.")
            } else {
                when (val state = profileState) {
                is PublicProfileState.Loading -> {
                    Box(modifier = Modifier.fillMaxWidth().padding(vertical = 48.dp), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(color = Fresh)
                    }
                }
                is PublicProfileState.Error -> {
                    EmptyStateCard(state.message, onRetry = { viewModel.fetchPublicProfile(resolvedHandle) })
                }
                is PublicProfileState.Success -> {
                    val profile = state.profile
                    CoverBanner(cover = profile.cover, brandColor = profile.brandColor, displayName = profile.displayName)
                    Spacer(modifier = Modifier.height(16.dp))

                    Text("@${profile.handle ?: "photographer"}", style = Typography.labelMedium, color = Fresh, fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(4.dp))
                    val sub = buildString {
                        profile.city?.takeIf { it.isNotBlank() }?.let { append(it); append("  ·  ") }
                        append("Member since ${profile.memberSince ?: "2026"}")
                    }
                    Text(sub, style = Typography.bodySmall, color = SlateSoft)

                    if (!profile.bio.isNullOrBlank()) {
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(profile.bio, style = Typography.bodyMedium, color = InkSoft)
                    }

                    // Stat row — web public-profile parity (Events / Photos /
                    // On QuickPitik; deliberately NO sales figure on a public
                    // surface).
                    Spacer(modifier = Modifier.height(20.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(28.dp)) {
                        StatNumber(
                            value = "${profile.events.size}",
                            label = "Events",
                        )
                        StatNumber(
                            value = "%,d".format(profile.events.sumOf { it.photoCount }),
                            label = "Photos",
                        )
                        StatNumber(
                            value = profile.memberSince ?: "2026",
                            label = "On QuickPitik",
                        )
                    }

                    Spacer(modifier = Modifier.height(24.dp))
                    Text("EVENTS COVERED", style = Typography.labelMedium, color = Slate)
                    Spacer(modifier = Modifier.height(12.dp))

                    if (profile.events.isEmpty()) {
                        Text("No public events yet.", style = Typography.bodyMedium, color = SlateSoft)
                    } else {
                        profile.events.forEach { coverage ->
                            EventCoverageCard(
                                coverage = coverage,
                                onClick = { selectedEventSlug = coverage.eventSlug }
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                        }
                    }
                }
            }
        }
    }
}
}

@Composable
private fun CoverBanner(cover: CoverSourceDto?, brandColor: String?, displayName: String?) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(170.dp)
            .clip(RoundedCornerShape(16.dp))
    ) {
        when {
            cover?.kind == "image" && !cover.url.isNullOrBlank() -> {
                AsyncImage(
                    model = resolveProfileImageUrl(cover.url),
                    contentDescription = "Cover",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            }
            cover?.kind == "gradient" -> {
                val from = parseHex(cover.from) ?: parseHex(brandColor) ?: Fresh
                val to = parseHex(cover.to) ?: Ink
                Box(modifier = Modifier.fillMaxSize().background(Brush.linearGradient(listOf(from, to))))
            }
            else -> {
                Box(modifier = Modifier.fillMaxSize().background(parseHex(brandColor) ?: Fresh))
            }
        }
        // Scrim for legible name
        Box(modifier = Modifier.fillMaxSize().background(WatermarkInk))
        Text(
            text = displayName ?: "Photographer",
            color = Color.White,
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.align(Alignment.BottomStart).padding(16.dp)
        )
    }
}

@Composable
private fun EventCoverageCard(coverage: PhotographerEventCoverageDto, onClick: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(BoneDeep)
            .border(BorderStroke(1.dp, Line), RoundedCornerShape(12.dp))
            .clickable { onClick() }
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(formatSlug(coverage.eventSlug), style = Typography.titleMedium, fontWeight = FontWeight.Bold, color = Ink, modifier = Modifier.weight(1f))
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(4.dp))
                    .background(Fresh.copy(alpha = 0.15f))
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            ) {
                Text(coverage.state.uppercase(), color = Fresh, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            }
        }
        Spacer(modifier = Modifier.height(6.dp))
        // No sales figure on a PUBLIC surface (web deliberately hides it —
        // this page is what runners see, including via the lightbox byline).
        Text(
            text = "${coverage.photoCount} photos",
            style = Typography.bodySmall,
            color = SlateSoft
        )
        Spacer(modifier = Modifier.height(8.dp))
        ArrowLabel("VIEW GALLERY →", color = Fresh, style = Typography.labelSmall, fontWeight = FontWeight.Bold, iconSize = 12.dp)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ProfileEventGalleryView(
    viewModel: PublicPhotographerViewModel,
    handle: String,
    slug: String,
    onBack: () -> Unit,
    cartViewModel: CartViewModel? = null,
) {
    val photosState by viewModel.profileEventPhotosState.collectAsState()
    val eventDetail by viewModel.galleryEventDetail.collectAsState()
    var selectedPhoto by remember { mutableStateOf<PhotoDto?>(null) }
    var bibFilter by rememberSaveable { mutableStateOf("") }
    var visiblePhotoLimit by rememberSaveable(slug) { mutableStateOf(20) }

    LaunchedEffect(slug) {
        viewModel.fetchProfileEventPhotos(handle, slug)
        // Real event name + the event id add-to-cart needs — the coverage row
        // only carries the slug.
        viewModel.fetchGalleryEventDetail(slug)
    }

    // Commerce needs a true-runner session, a cart, and the event id.
    val commerce = cartViewModel != null && rememberIsTrueRunner() && eventDetail != null
    val cartItems = cartViewModel?.cartItems?.collectAsState()?.value.orEmpty()

    Surface(modifier = Modifier.fillMaxSize(), color = Bone) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(20.dp)
        ) {
            BackRow(label = "BACK TO PROFILE", onBack = onBack)
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = eventDetail?.name ?: formatSlug(slug),
                style = Typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = Ink,
            )
            Text("@$handle", style = Typography.bodyMedium, color = SlateSoft)
            Spacer(modifier = Modifier.height(12.dp))
            // Client-side bib filter — web /{handle}/events/[slug] parity
            // (that page also filters the loaded set client-side).
            TextField(
                value = bibFilter,
                onValueChange = {
                    bibFilter = it
                    visiblePhotoLimit = 20
                },
                placeholder = { Text("Filter by bib number…", color = SlateSoft) },
                singleLine = true,
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = BoneDeep,
                    unfocusedContainerColor = BoneDeep,
                    focusedIndicatorColor = Fresh,
                    unfocusedIndicatorColor = Color.Transparent,
                    focusedTextColor = Ink,
                    unfocusedTextColor = InkSoft,
                ),
                shape = FieldShape,
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(modifier = Modifier.height(20.dp))

            when (val state = photosState) {
                is ProfileEventPhotosState.Loading -> {
                    Box(modifier = Modifier.fillMaxWidth().padding(vertical = 32.dp), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(color = Fresh)
                    }
                }
                is ProfileEventPhotosState.Error -> {
                    Box(modifier = Modifier.fillMaxWidth().padding(vertical = 32.dp), contentAlignment = Alignment.Center) {
                        Text(state.message, color = ErrorRed, textAlign = TextAlign.Center, style = Typography.bodyMedium)
                    }
                }
                is ProfileEventPhotosState.Success -> {
                    val visiblePhotos = remember(state.photos, bibFilter) {
                        val q = bibFilter.trim()
                        if (q.isEmpty()) state.photos
                        else state.photos.filter { it.bib?.contains(q, ignoreCase = true) == true }
                    }
                    if (visiblePhotos.isEmpty()) {
                        Box(modifier = Modifier.fillMaxWidth().padding(vertical = 32.dp), contentAlignment = Alignment.Center) {
                            Text(
                                text = if (bibFilter.isBlank()) "No photos in this gallery yet."
                                else "No photos for bib ${bibFilter.trim()} in this gallery.",
                                color = SlateSoft,
                                textAlign = TextAlign.Center,
                                style = Typography.bodyMedium,
                            )
                        }
                    } else {
                        visiblePhotos.take(visiblePhotoLimit).chunked(2).forEach { rowPhotos ->
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                rowPhotos.forEach { photo ->
                                    Box(
                                        modifier = Modifier
                                            .weight(1f)
                                            .aspectRatio(0.85f)
                                            .clip(RoundedCornerShape(16.dp))
                                            .background(BoneDeep)
                                            .clickable { selectedPhoto = photo },
                                        contentAlignment = Alignment.Center
                                    ) {
                                        val url = resolveProfileImageUrl(photo.imageUrl)
                                        if (url != null) {
                                            AsyncImage(
                                                model = url,
                                                contentDescription = photo.alt ?: "Photo",
                                                contentScale = ContentScale.Crop,
                                                modifier = Modifier.fillMaxSize()
                                            )
                                        }
                                        Box(
                                            modifier = Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.04f)),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Text(
                                                text = "QUICKPITIK\nPREVIEW",
                                                color = Color.White.copy(alpha = 0.35f),
                                                fontSize = 11.sp,
                                                fontWeight = FontWeight.Bold,
                                                textAlign = TextAlign.Center
                                            )
                                        }
                                    }
                                }
                                if (rowPhotos.size == 1) {
                                    Spacer(modifier = Modifier.weight(1f))
                                }
                            }
                            Spacer(modifier = Modifier.height(12.dp))
                        }
                        if (visiblePhotoLimit < visiblePhotos.size) {
                            GhostCta(
                                text = "Load more",
                                onClick = { visiblePhotoLimit += 20 },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                    }
                }
            }
        }
    }

    // The shared runner lightbox replaces the old 220dp AlertDialog "Price:
    // ₱X / CLOSE" preview — with a pager, the photographer credit suppressed
    // (we're on their page), and, for a true runner with the event id loaded,
    // the full Add-to-cart / Buy-now flow. This is the web's transactional
    // /{handle}/events/[slug] gallery, not just a viewer.
    val preview = selectedPhoto
    val allPhotos = (photosState as? ProfileEventPhotosState.Success)?.photos.orEmpty()
    if (preview != null && allPhotos.isNotEmpty()) {
        val previewPhotos = allPhotos.map { it.toPreviewData(eventDetail?.name ?: formatSlug(slug)) }
        val currentIndex = previewPhotos.indexOfFirst { it.id == preview.id }
        if (currentIndex >= 0) {
            PhotoPreview(
                photos = previewPhotos,
                currentIndex = currentIndex,
                commerceEnabled = commerce,
                isInCart = { data -> cartItems.any { it.photoId == data.id } },
                onToggleCart = onToggleCart@{ data ->
                    val cart = cartViewModel ?: return@onToggleCart
                    val detail = eventDetail ?: return@onToggleCart
                    val photo = allPhotos.firstOrNull { it.id == data.id } ?: return@onToggleCart
                    if (cartItems.any { it.photoId == data.id }) {
                        cart.removeFromCart(data.id)
                    } else {
                        cart.addToCart(photo, detail.id, detail.slug, detail.name)
                    }
                },
                onBuyNow = onBuyNow@{ data ->
                    val cart = cartViewModel ?: return@onBuyNow
                    val detail = eventDetail ?: return@onBuyNow
                    val photo = allPhotos.firstOrNull { it.id == data.id } ?: return@onBuyNow
                    if (cartItems.any { it.photoId == data.id }) {
                        cart.openCheckoutSheet()
                    } else {
                        cart.triggerExpressCheckout()
                        cart.addToCart(photo, detail.id, detail.slug, detail.name)
                    }
                    selectedPhoto = null
                },
                onClose = { selectedPhoto = null },
                onIndexChange = { newIndex ->
                    allPhotos.getOrNull(newIndex)?.let { selectedPhoto = it }
                },
            )
        }
    }
}

@Composable
private fun BackRow(label: String, onBack: () -> Unit) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .clickable { onBack() }
            .padding(vertical = 6.dp, horizontal = 2.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = Slate)
        Spacer(modifier = Modifier.width(8.dp))
        Text(label, style = Typography.labelMedium, color = Slate, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun EmptyStateCard(message: String, onRetry: (() -> Unit)? = null) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(BoneDeep)
            .border(BorderStroke(1.dp, Line), RoundedCornerShape(16.dp))
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(message, color = SlateSoft, textAlign = TextAlign.Center, style = Typography.bodyMedium)
        if (onRetry != null) {
            Spacer(modifier = Modifier.height(16.dp))
            Button(
                onClick = onRetry,
                colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                shape = RoundedCornerShape(8.dp)
            ) {
                Text("RETRY", fontWeight = FontWeight.Bold)
            }
        }
    }
}

private fun formatSlug(slug: String): String =
    slug.split("-").filter { it.isNotBlank() }.joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }

private fun parseHex(hex: String?): Color? {
    if (hex.isNullOrBlank()) return null
    return try {
        Color(AndroidColor.parseColor(if (hex.startsWith("#")) hex else "#$hex"))
    } catch (e: Exception) {
        null
    }
}

private fun resolveProfileImageUrl(url: String?): String? =
    RetrofitClient.resolveImageUrl(url?.takeIf { it.isNotBlank() })
