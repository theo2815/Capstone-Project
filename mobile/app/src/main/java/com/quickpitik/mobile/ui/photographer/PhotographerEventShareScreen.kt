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
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
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
import com.quickpitik.mobile.data.download.PhotoDownloader
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.data.remote.PhotographerLibraryPhotoDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.runner.PhotoPreview
import com.quickpitik.mobile.ui.runner.PhotoPreviewData
import com.quickpitik.mobile.ui.runner.PhotoPreviewMode
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.BrandLogo
import com.quickpitik.mobile.ui.theme.ErrorView
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.StatNumber
import com.quickpitik.mobile.ui.theme.TileShape
import com.quickpitik.mobile.ui.theme.Typography
import kotlinx.coroutines.launch
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
    var visiblePhotoLimit by rememberSaveable(event.id) { mutableStateOf(20) }
    val scope = rememberCoroutineScope()
    // Guards the lightbox download: the presigned-URL fetch plus the save is a
    // multi-second round trip, and repeat taps would queue duplicate saves.
    var downloading by remember { mutableStateOf(false) }

    LaunchedEffect(event.id) {
        viewModel.fetchSharePhotos(event.id)
        // The handle is read from the shared brandSettings flow, hydrated by
        // the VM's init + Settings refreshes; re-fetch only the one payload this
        // screen needs if it never loaded.
        if (viewModel.brandSettings.value == null) viewModel.fetchBrandSettings()
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
            // Back + brand
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Row(
                    modifier = Modifier
                        .clip(TileShape)
                        .clickable { onBack() }
                        .padding(vertical = 6.dp, horizontal = 2.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = Slate)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("BACK TO EVENTS", style = Typography.labelMedium, color = Slate, fontWeight = FontWeight.Bold)
                }
                BrandLogo(compact = true)
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
                        .clip(QpCardShape)
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
            QpCard(
                modifier = Modifier.fillMaxWidth(),
                padding = 20.dp
            ) {
                Kicker("Public gallery")
                Spacer(modifier = Modifier.height(12.dp))
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(PillShape)
                        .background(Bone)
                        .border(BorderStroke(1.dp, Line), PillShape)
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
                    PrimaryCta(
                        text = "Copy link",
                        onClick = {
                            copyLink(context, fullUrl)
                            Toast.makeText(context, "Link copied.", Toast.LENGTH_SHORT).show()
                        },
                        modifier = Modifier.weight(1f)
                    )
                    GhostCta(
                        text = "Share",
                        onClick = { shareNative(context, event.name, fullUrl) },
                        modifier = Modifier.weight(1f)
                    )
                }
                Spacer(modifier = Modifier.height(18.dp))
                Kicker("Share to your followers", color = SlateSoft)
                Spacer(modifier = Modifier.height(10.dp))
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    ShareChip(label = "Facebook", dotColor = FacebookBlue) {
                        openUrl(context, "https://www.facebook.com/sharer/sharer.php?u=${enc(fullUrl)}")
                    }
                    ShareChip(label = "Instagram", dotColor = InstagramPink) {
                        copyLink(context, fullUrl)
                        Toast.makeText(context, "Link copied. Paste into your IG bio or story.", Toast.LENGTH_LONG).show()
                    }
                    ShareChip(label = "X", dotColor = XBlack) {
                        openUrl(context, "https://twitter.com/intent/tweet?url=${enc(fullUrl)}&text=${enc("Photos from ${event.name}")}")
                    }
                    ShareChip(label = "Threads", dotColor = ThreadsGray) {
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
                StatNumber(value = "${event.photoCount}", label = "Photos")
                StatNumber(value = "${event.salesCount}", label = "Sold")
                StatNumber(value = "₱%,.0f".format(event.revenueKept), label = "Kept", valueColor = Fresh)
            }
            Spacer(modifier = Modifier.height(24.dp))

            // Mosaic
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Kicker("Uploaded photos")
            }
            Spacer(modifier = Modifier.height(12.dp))

            when (val state = photosState) {
                is SharePhotosState.Loading -> {
                    Box(modifier = Modifier.fillMaxWidth().padding(vertical = 32.dp), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(color = Fresh)
                    }
                }
                is SharePhotosState.Error -> {
                    ErrorView(
                        message = state.message,
                        title = "Couldn't load your uploads",
                        onRetry = { viewModel.fetchSharePhotos(event.id) },
                        modifier = Modifier.fillMaxWidth(),
                    )
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
                        state.photos.take(visiblePhotoLimit).chunked(2).forEachIndexed { rowIdx, rowPhotos ->
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
                        if (visiblePhotoLimit < state.photos.size) {
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
            // Two hops, because the library listing only carries the
            // watermarked thumbnail: resolve the presigned original, then hand
            // it to the same saver the runner order screens use.
            onDownload = { photo ->
                if (!downloading) {
                    downloading = true
                    scope.launch {
                        val result = viewModel.resolvePhotoDownloadUrl(photo.id)
                        val message = result.fold(
                            onSuccess = { url ->
                                // PhotoPreviewData carries no bib, but the DTO
                                // list behind it does — look it up so the saved
                                // file is bib-tagged like the website's.
                                val bib = current.photos.firstOrNull { it.id == photo.id }?.bib
                                val filename = PhotoDownloader.buildFilename(photo.id, bib)
                                when (val saved = PhotoDownloader.saveToGallery(context, url, filename)) {
                                    is PhotoDownloader.Result.Saved -> "Saved ${saved.displayName} to your gallery."
                                    is PhotoDownloader.Result.Failed -> saved.message
                                }
                            },
                            onFailure = { it.message ?: "Couldn't get the download link." },
                        )
                        Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                        downloading = false
                    }
                }
            },
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
            .clip(QpCardShape)
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
                    .clip(PillShape)
                    .background(Ink.copy(alpha = 0.55f))
                    .padding(horizontal = 8.dp, vertical = 3.dp)
            ) {
                Text("${photo.salesCount} sold", color = Bone, fontSize = 9.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun ShareChip(label: String, dotColor: Color, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .clip(PillShape)
            .background(Bone)
            .border(BorderStroke(1.dp, Line), PillShape)
            .clickable { onClick() }
            .padding(horizontal = 14.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Box(modifier = Modifier.size(8.dp).clip(PillShape).background(dotColor))
        Text(label, style = Typography.bodyMedium, color = Ink, fontWeight = FontWeight.Medium)
    }
}

// Platform brand colors for the share-chip dots — external identities, not
// theme tokens, but named so they don't read as stray magic hexes.
private val FacebookBlue = Color(0xFF1877F2)
private val InstagramPink = Color(0xFFE4405F)
private val XBlack = Color(0xFF000000)
private val ThreadsGray = Color(0xFF444444)

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

private fun resolveShareImageUrl(url: String?): String? =
    RetrofitClient.resolveImageUrl(url?.takeIf { it.isNotBlank() })
