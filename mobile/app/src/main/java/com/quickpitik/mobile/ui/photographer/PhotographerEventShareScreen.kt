package com.quickpitik.mobile.ui.photographer

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.data.remote.PhotographerLibraryPhotoDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.runner.PhotoPreview
import com.quickpitik.mobile.ui.runner.PhotoPreviewData
import com.quickpitik.mobile.ui.runner.PhotoPreviewMode
import com.quickpitik.mobile.ui.theme.*
import java.net.URLEncoder

private fun PhotographerLibraryPhotoDto.toOwnerPreviewData(eventName: String?): PhotoPreviewData =
    PhotoPreviewData(
        id = id,
        price = 0.0,
        imageUrl = resolveShareImageUrl(thumbnailUrl),
        eventName = eventName,
        salesCount = salesCount,
    )

// Focused share page — mobile mirror of website /dashboard/events/[id].
// Layout: back → hero → public-gallery share band (copy + native share + social
// chips) → stats → uploaded-photo mosaic. Photos come from the photographer-
// scoped GET /me/photographer/events/{id}/photos, so the grid shows only this
// photographer's uploads (not the whole event).
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun PhotographerEventShareScreen(
    event: PhotographerEventSummaryDto,
    viewModel: PhotographerDashboardViewModel,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val brand by viewModel.brandSettings.collectAsState()
    val photosState by viewModel.sharePhotosState.collectAsState()
    var selectedIndex by remember { mutableStateOf<Int?>(null) }

    LaunchedEffect(event.id) {
        viewModel.fetchSharePhotos(event.id)
        viewModel.fetchSettings()
    }

    val handle = brand?.handle?.takeIf { it.isNotBlank() } ?: "your-handle"
    val displayUrl = "quickpitik.com/$handle/events/${event.slug}"
    val fullUrl = "https://$displayUrl"

    Surface(modifier = Modifier.fillMaxSize(), color = Bone) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(20.dp)
        ) {
            // Back
            Row(
                modifier = Modifier
                    .clip(RoundedCornerShape(8.dp))
                    .clickable { onBack() }
                    .padding(vertical = 6.dp, horizontal = 2.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = Slate)
                Spacer(modifier = Modifier.width(8.dp))
                Text("BACK TO EVENTS", style = Typography.labelMedium, color = Slate, fontWeight = FontWeight.Bold)
            }
            Spacer(modifier = Modifier.height(20.dp))

            // Hero
            val banner = resolveShareImageUrl(event.bannerUrl)
            if (banner != null) {
                AsyncImage(
                    model = banner,
                    contentDescription = "Event banner",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(180.dp)
                        .clip(RoundedCornerShape(16.dp))
                )
                Spacer(modifier = Modifier.height(16.dp))
            }
            Text(
                text = "${event.date}  ·  ${event.state.uppercase()}",
                style = Typography.labelSmall,
                color = Slate
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(event.name, style = Typography.titleLarge, fontWeight = FontWeight.Bold, color = Ink)
            Spacer(modifier = Modifier.height(6.dp))
            Text(event.location, style = Typography.bodyMedium, color = InkSoft)
            Spacer(modifier = Modifier.height(24.dp))

            // Share band
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(BoneDeep)
                    .border(BorderStroke(1.dp, Line), RoundedCornerShape(16.dp))
                    .padding(20.dp)
            ) {
                Text("PUBLIC GALLERY", style = Typography.labelSmall, color = Slate)
                Spacer(modifier = Modifier.height(12.dp))
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(100))
                        .background(Bone)
                        .border(BorderStroke(1.dp, Line), RoundedCornerShape(100))
                        .padding(horizontal = 16.dp, vertical = 12.dp)
                ) {
                    Text(
                        text = displayUrl,
                        style = Typography.bodyMedium,
                        color = Ink,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                Spacer(modifier = Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                    Button(
                        onClick = {
                            copyLink(context, fullUrl)
                            Toast.makeText(context, "Link copied.", Toast.LENGTH_SHORT).show()
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                        shape = RoundedCornerShape(100),
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("COPY LINK", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                    }
                    OutlinedButton(
                        onClick = { shareNative(context, event.name, fullUrl) },
                        border = BorderStroke(1.dp, Ink),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink),
                        shape = RoundedCornerShape(100),
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("SHARE", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                    }
                }
                Spacer(modifier = Modifier.height(18.dp))
                Text("SHARE TO YOUR FOLLOWERS", style = Typography.labelSmall, color = SlateSoft)
                Spacer(modifier = Modifier.height(10.dp))
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    ShareChip(label = "Facebook", dotColor = Color(0xFF1877F2)) {
                        openUrl(context, "https://www.facebook.com/sharer/sharer.php?u=${enc(fullUrl)}")
                    }
                    ShareChip(label = "Instagram", dotColor = Color(0xFFE4405F)) {
                        copyLink(context, fullUrl)
                        Toast.makeText(context, "Link copied. Paste into your IG bio or story.", Toast.LENGTH_LONG).show()
                    }
                    ShareChip(label = "X", dotColor = Color(0xFF000000)) {
                        openUrl(context, "https://twitter.com/intent/tweet?url=${enc(fullUrl)}&text=${enc("Photos from ${event.name}")}")
                    }
                    ShareChip(label = "Threads", dotColor = Color(0xFF444444)) {
                        openUrl(context, "https://www.threads.net/intent/post?text=${enc("Photos from ${event.name} — $displayUrl")}")
                    }
                }
            }
            Spacer(modifier = Modifier.height(24.dp))

            // Stats
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(24.dp)
            ) {
                ShareStat(value = "${event.photoCount}", label = "PHOTOS")
                ShareStat(value = "${event.salesCount}", label = "SOLD")
                ShareStat(value = "₱%,.0f".format(event.revenueKept), label = "KEPT", valueColor = Fresh)
            }
            Spacer(modifier = Modifier.height(24.dp))

            // Mosaic
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("UPLOADED PHOTOS", style = Typography.labelMedium, color = Slate)
            }
            Spacer(modifier = Modifier.height(12.dp))

            when (val state = photosState) {
                is SharePhotosState.Loading -> {
                    Box(modifier = Modifier.fillMaxWidth().padding(vertical = 32.dp), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(color = Fresh)
                    }
                }
                is SharePhotosState.Error -> {
                    Box(modifier = Modifier.fillMaxWidth().padding(vertical = 32.dp), contentAlignment = Alignment.Center) {
                        Text(state.message, color = ErrorRed, textAlign = TextAlign.Center, style = Typography.bodyMedium)
                    }
                }
                is SharePhotosState.Success -> {
                    if (state.photos.isEmpty()) {
                        Box(modifier = Modifier.fillMaxWidth().padding(vertical = 32.dp), contentAlignment = Alignment.Center) {
                            Text(
                                "No uploaded photos for this event yet.",
                                color = SlateSoft,
                                textAlign = TextAlign.Center,
                                style = Typography.bodyMedium
                            )
                        }
                    } else {
                        state.photos.chunked(2).forEachIndexed { rowIdx, rowPhotos ->
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                rowPhotos.forEachIndexed { colIdx, photo ->
                                    val flatIdx = rowIdx * 2 + colIdx
                                    SharePhotoTile(
                                        photo = photo,
                                        modifier = Modifier.weight(1f),
                                        onClick = { selectedIndex = flatIdx }
                                    )
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

    // Reuses the runner-side PhotoPreview lightbox in OwnerReview mode — swipe
    // through all uploaded photos, no purchase UI. Same Dialog/Pager visual as
    // the runner browse flow, so the photographer sees their work in the
    // primary lightbox experience (not a cramped AlertDialog).
    val idx = selectedIndex
    val current = photosState
    if (idx != null && current is SharePhotosState.Success && current.photos.isNotEmpty()) {
        val previewPhotos = remember(current.photos, event.name) {
            current.photos.map { it.toOwnerPreviewData(event.name) }
        }
        val safeIdx = idx.coerceIn(0, previewPhotos.size - 1)
        PhotoPreview(
            photos = previewPhotos,
            currentIndex = safeIdx,
            onClose = { selectedIndex = null },
            onIndexChange = { selectedIndex = it },
            mode = PhotoPreviewMode.OwnerReview,
        )
    }
}

@Composable
private fun SharePhotoTile(
    photo: PhotographerLibraryPhotoDto,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Box(
        modifier = modifier
            .aspectRatio(0.85f)
            .clip(RoundedCornerShape(16.dp))
            .background(BoneDeep)
            .clickable { onClick() },
        contentAlignment = Alignment.Center
    ) {
        val url = resolveShareImageUrl(photo.thumbnailUrl)
        if (url != null) {
            AsyncImage(
                model = url,
                contentDescription = "Uploaded photo",
                contentScale = ContentScale.Crop,
                modifier = Modifier.fillMaxSize()
            )
        }
        if (photo.salesCount > 0) {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(8.dp)
                    .clip(RoundedCornerShape(100))
                    .background(Ink.copy(alpha = 0.55f))
                    .padding(horizontal = 8.dp, vertical = 3.dp)
            ) {
                Text("${photo.salesCount} sold", color = Bone, fontSize = 9.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun ShareStat(value: String, label: String, valueColor: Color = Ink) {
    Column {
        Text(value, color = valueColor, fontSize = 18.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(2.dp))
        Text(label, color = SlateSoft, fontSize = 9.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.5.sp)
    }
}

@Composable
private fun ShareChip(label: String, dotColor: Color, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(100))
            .background(Bone)
            .border(BorderStroke(1.dp, Line), RoundedCornerShape(100))
            .clickable { onClick() }
            .padding(horizontal = 14.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Box(modifier = Modifier.size(8.dp).clip(RoundedCornerShape(100)).background(dotColor))
        Text(label, style = Typography.bodyMedium, color = Ink, fontWeight = FontWeight.Medium)
    }
}

private fun enc(value: String): String = URLEncoder.encode(value, "UTF-8")

private fun copyLink(context: Context, url: String) {
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    clipboard.setPrimaryClip(ClipData.newPlainText("QuickPitik gallery", url))
}

private fun shareNative(context: Context, eventName: String, url: String) {
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "text/plain"
        putExtra(Intent.EXTRA_SUBJECT, eventName)
        putExtra(Intent.EXTRA_TEXT, "Photos from $eventName — $url")
    }
    context.startActivity(Intent.createChooser(send, "Share gallery"))
}

private fun openUrl(context: Context, url: String) {
    context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
}

private fun resolveShareImageUrl(url: String?): String? {
    if (url.isNullOrBlank()) return null
    // M-2 (2026-05-27 PM): host derived from RetrofitClient.BASE_URL.
    if (url.startsWith("/")) return "${RetrofitClient.backendOrigin}$url"
    return url.replace("localhost", RetrofitClient.backendHost)
        .replace("127.0.0.1", RetrofitClient.backendHost)
}
