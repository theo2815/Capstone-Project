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
import com.quickpitik.mobile.ui.theme.*

// Mobile mirror of website /{handle} (public photographer profile) and
// /{handle}/events/[slug] (per-event public gallery). Reached via "Preview
// public profile" in the photographer Overview tab — the only handle the app
// knows without a backend change is the logged-in photographer's own.
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotographerPublicProfileScreen(
    viewModel: PhotographerDashboardViewModel,
    onBack: () -> Unit
) {
    val brand by viewModel.brandSettings.collectAsState()
    val profileState by viewModel.publicProfileState.collectAsState()
    val handle = brand?.handle?.takeIf { it.isNotBlank() }
    var selectedEventSlug by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(handle) {
        if (handle == null) viewModel.fetchSettings() else viewModel.fetchPublicProfile(handle)
    }

    val gallerySlug = selectedEventSlug
    if (handle != null && gallerySlug != null) {
        ProfileEventGalleryView(
            viewModel = viewModel,
            handle = handle,
            slug = gallerySlug,
            onBack = { selectedEventSlug = null }
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

            if (handle == null) {
                EmptyStateCard("Set your public handle in the Settings tab to preview your profile.")
                return@Column
            }

            when (val state = profileState) {
                is PublicProfileState.Loading -> {
                    Box(modifier = Modifier.fillMaxWidth().padding(vertical = 48.dp), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(color = Fresh)
                    }
                }
                is PublicProfileState.Error -> {
                    EmptyStateCard(state.message, onRetry = { viewModel.fetchPublicProfile(handle) })
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
        Box(modifier = Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.28f)))
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
        Text(
            text = "${coverage.photoCount} photos  ·  ${coverage.salesCount} sold",
            style = Typography.bodySmall,
            color = SlateSoft
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text("VIEW GALLERY →", style = Typography.labelSmall, color = Fresh, fontWeight = FontWeight.Bold)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ProfileEventGalleryView(
    viewModel: PhotographerDashboardViewModel,
    handle: String,
    slug: String,
    onBack: () -> Unit
) {
    val photosState by viewModel.profileEventPhotosState.collectAsState()
    var selectedPhoto by remember { mutableStateOf<PhotoDto?>(null) }

    LaunchedEffect(slug) { viewModel.fetchProfileEventPhotos(handle, slug) }

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
            Text(formatSlug(slug), style = Typography.titleLarge, fontWeight = FontWeight.Bold, color = Ink)
            Text("@$handle", style = Typography.bodyMedium, color = SlateSoft)
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
                    if (state.photos.isEmpty()) {
                        Box(modifier = Modifier.fillMaxWidth().padding(vertical = 32.dp), contentAlignment = Alignment.Center) {
                            Text("No photos in this gallery yet.", color = SlateSoft, textAlign = TextAlign.Center, style = Typography.bodyMedium)
                        }
                    } else {
                        state.photos.chunked(2).forEach { rowPhotos ->
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
                    }
                }
            }
        }
    }

    val preview = selectedPhoto
    if (preview != null) {
        AlertDialog(
            onDismissRequest = { selectedPhoto = null },
            title = { Text("Photo", fontWeight = FontWeight.Bold, color = Ink) },
            text = {
                Column(modifier = Modifier.fillMaxWidth()) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(220.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .background(BoneDeep),
                        contentAlignment = Alignment.Center
                    ) {
                        val url = resolveProfileImageUrl(preview.imageUrl)
                        if (url != null) {
                            AsyncImage(model = url, contentDescription = "Preview", contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
                        }
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    Text("Bib: ${preview.bib ?: "N/A"}", style = Typography.bodyMedium, color = Ink)
                    Text("Time: ${preview.time}", style = Typography.bodyMedium, color = Ink)
                    Text(String.format("Price: ₱%,.2f", preview.price), style = Typography.bodyMedium, color = Ink)
                }
            },
            confirmButton = {
                TextButton(onClick = { selectedPhoto = null }) {
                    Text("CLOSE", color = Ink, fontWeight = FontWeight.Bold)
                }
            },
            containerColor = Bone,
            shape = RoundedCornerShape(16.dp)
        )
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

private fun resolveProfileImageUrl(url: String?): String? {
    if (url.isNullOrBlank()) return null
    if (url.startsWith("/")) return "http://10.0.2.2:8080$url"
    return url.replace("localhost", "10.0.2.2").replace("127.0.0.1", "10.0.2.2")
}
