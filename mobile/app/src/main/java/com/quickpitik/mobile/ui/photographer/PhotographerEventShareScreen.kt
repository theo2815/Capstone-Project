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
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import androidx.compose.ui.draw.alpha
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
import com.quickpitik.mobile.ui.common.PhotoPreview
import com.quickpitik.mobile.ui.common.PhotoPreviewMode
import com.quickpitik.mobile.ui.common.peso
import com.quickpitik.mobile.ui.common.toOwnerPreviewData
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.BrandLogo
import com.quickpitik.mobile.ui.theme.ErrorRed
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
import com.quickpitik.mobile.ui.theme.SurfaceWhite
import com.quickpitik.mobile.ui.theme.TileShape
import com.quickpitik.mobile.ui.theme.Typography
import kotlinx.coroutines.launch
import java.net.URLEncoder

private enum class PhotoSection { Active, Deleted }

// Focused share page — mobile mirror of website /dashboard/events/[id].
// Layout: back → hero → public-gallery share band → stats → photo library.
// The library carries Active + Deleted (trash) tabs and a touch-friendly
// Select mode (bulk trash / restore / permanent delete), plus "Download all".
// Photos come from the photographer-scoped GET /me/photographer/events/{id}/
// photos, so the grid shows only this photographer's uploads.
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun PhotographerEventShareScreen(
    event: PhotographerEventSummaryDto,
    viewModel: PhotographerDashboardViewModel,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val activeState by viewModel.sharePhotosState.collectAsState()
    val trashState by viewModel.trashPhotosState.collectAsState()

    var section by remember(event.id) { mutableStateOf(PhotoSection.Active) }
    var selectedIndex by remember { mutableStateOf<Int?>(null) }
    var visiblePhotoLimit by rememberSaveable(event.id) { mutableStateOf(20) }
    // Selection is transient screen UI state (reset on tab switch), not VM data.
    var selecting by remember(event.id) { mutableStateOf(false) }
    var selected by remember(event.id) { mutableStateOf<Set<String>>(emptySet()) }
    var busy by remember { mutableStateOf(false) }
    var confirmPermanent by remember { mutableStateOf(false) }
    var downloadingAll by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    // Guards the lightbox download: the presigned-URL fetch plus the save is a
    // multi-second round trip, and repeat taps would queue duplicate saves.
    var downloading by remember { mutableStateOf(false) }

    LaunchedEffect(event.id) {
        viewModel.fetchSharePhotos(event.id)
    }
    // The trash list is only fetched when its tab is first opened.
    LaunchedEffect(section, event.id) {
        if (section == PhotoSection.Deleted) viewModel.fetchTrashPhotos(event.id)
    }

    val currentState = if (section == PhotoSection.Active) activeState else trashState
    val currentPhotos = (currentState as? SharePhotosState.Success)?.photos ?: emptyList()

    fun switchSection(next: PhotoSection) {
        if (next == section) return
        section = next
        selecting = false
        selected = emptySet()
        selectedIndex = null
        visiblePhotoLimit = 20
    }

    fun toggle(id: String) {
        selected = if (id in selected) selected - id else selected + id
    }

    fun exitSelect() {
        selecting = false
        selected = emptySet()
    }

    // The gallery URL is built from the caller's handle on the event payload
    // (backend truth) — not the settings flow, which could still hold a blank
    // placeholder. No handle yet ⇒ no shareable link, so the share actions stay
    // disabled instead of copying a broken quickpitik.com/your-handle/… link.
    val handle = event.handle?.takeIf { it.isNotBlank() }
    val hasHandle = handle != null
    val displayUrl = if (hasHandle) {
        "quickpitik.com/$handle/events/${event.slug}"
    } else {
        "Set a handle in Settings to share this gallery"
    }
    val fullUrl = if (hasHandle) "https://quickpitik.com/$handle/events/${event.slug}" else ""

    Box(modifier = Modifier.fillMaxSize()) {
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
                            enabled = hasHandle,
                            modifier = Modifier.weight(1f)
                        )
                        GhostCta(
                            text = "Share",
                            onClick = { shareNative(context, event.name, fullUrl) },
                            enabled = hasHandle,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Spacer(modifier = Modifier.height(18.dp))
                    Kicker("Share to your followers", color = SlateSoft)
                    Spacer(modifier = Modifier.height(10.dp))
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        ShareChip(label = "Facebook", dotColor = FacebookBlue, enabled = hasHandle) {
                            openUrl(context, "https://www.facebook.com/sharer/sharer.php?u=${enc(fullUrl)}")
                        }
                        ShareChip(label = "Instagram", dotColor = InstagramPink, enabled = hasHandle) {
                            copyLink(context, fullUrl)
                            Toast.makeText(context, "Link copied. Paste into your IG bio or story.", Toast.LENGTH_LONG).show()
                        }
                        ShareChip(label = "X", dotColor = XBlack, enabled = hasHandle) {
                            openUrl(context, "https://twitter.com/intent/tweet?url=${enc(fullUrl)}&text=${enc("Photos from ${event.name}")}")
                        }
                        ShareChip(label = "Threads", dotColor = ThreadsGray, enabled = hasHandle) {
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
                    StatNumber(value = peso(event.revenueKept, 0), label = "Kept", valueColor = Fresh)
                }
                Spacer(modifier = Modifier.height(24.dp))

                // Library tabs + actions
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    LibraryTabs(section = section, onSelect = ::switchSection)
                    if (!selecting && currentPhotos.isNotEmpty()) {
                        HeaderPill(text = "Select", onClick = { selecting = true })
                    }
                }

                // Download all (active tab, live gallery only)
                if (section == PhotoSection.Active && !selecting) {
                    val hasActive = (activeState as? SharePhotosState.Success)?.photos?.isNotEmpty() == true
                    if (hasActive) {
                        Spacer(modifier = Modifier.height(12.dp))
                        GhostCta(
                            text = if (downloadingAll) "Preparing…" else "Download all photos",
                            enabled = !downloadingAll,
                            modifier = Modifier.fillMaxWidth(),
                            onClick = {
                                if (downloadingAll) return@GhostCta
                                downloadingAll = true
                                scope.launch {
                                    val message = viewModel.prepareBundleUrl(event.id).fold(
                                        onSuccess = { url ->
                                            when (val saved = PhotoDownloader.saveBundleToDownloads(context, url)) {
                                                is PhotoDownloader.Result.Saved -> "Saved ${saved.displayName} to Downloads."
                                                is PhotoDownloader.Result.Failed -> saved.message
                                            }
                                        },
                                        onFailure = { it.message ?: "Couldn't start the download." },
                                    )
                                    Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                                    downloadingAll = false
                                }
                            },
                        )
                    }
                }
                Spacer(modifier = Modifier.height(16.dp))

                if (currentPhotos.any { it.deletionStartedAt != null }) {
                    Text(
                        "Some photos have started permanent deletion and cannot be restored. Select them and retry Delete forever to finish cleanup.",
                        style = Typography.bodyMedium,
                        color = Slate,
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                }

                when (val state = currentState) {
                    is SharePhotosState.Loading -> {
                        Box(modifier = Modifier.fillMaxWidth().padding(vertical = 32.dp), contentAlignment = Alignment.Center) {
                            CircularProgressIndicator(color = Fresh)
                        }
                    }
                    is SharePhotosState.Error -> {
                        ErrorView(
                            message = state.message,
                            title = if (section == PhotoSection.Active) "Couldn't load your uploads" else "Couldn't load Deleted",
                            onRetry = {
                                if (section == PhotoSection.Active) viewModel.fetchSharePhotos(event.id)
                                else viewModel.fetchTrashPhotos(event.id)
                            },
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                    is SharePhotosState.Success -> {
                        if (state.photos.isEmpty()) {
                            Box(modifier = Modifier.fillMaxWidth().padding(vertical = 32.dp), contentAlignment = Alignment.Center) {
                                Text(
                                    text = if (section == PhotoSection.Active) {
                                        "No uploaded photos for this event yet."
                                    } else {
                                        "Nothing in Deleted. Photos you delete land here first — restore them any time, or remove them for good."
                                    },
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
                                            selecting = selecting,
                                            selected = photo.id in selected,
                                            modifier = Modifier.weight(1f),
                                            onClick = {
                                                if (selecting) toggle(photo.id)
                                                else if (photo.deletionStartedAt != null) {
                                                    selecting = true
                                                    toggle(photo.id)
                                                } else selectedIndex = flatIdx
                                            }
                                        )
                                    }
                                    if (rowPhotos.size == 1) {
                                        Spacer(modifier = Modifier.weight(1f))
                                    }
                                }
                                Spacer(modifier = Modifier.height(12.dp))
                            }
                            if (state.loadMoreError != null) {
                                Text(state.loadMoreError, style = Typography.bodyMedium, color = ErrorRed)
                            }
                            if (visiblePhotoLimit < state.photos.size || state.hasMore) {
                                GhostCta(
                                    text = if (state.loadingMore) "Loading…" else if (state.loadMoreError != null) "Retry load more" else "Load more",
                                    enabled = !state.loadingMore,
                                    onClick = {
                                        if (visiblePhotoLimit >= state.photos.size) {
                                            if (section == PhotoSection.Deleted) viewModel.fetchTrashPhotos(event.id, loadMore = true)
                                            else viewModel.fetchSharePhotos(event.id, loadMore = true)
                                        }
                                        visiblePhotoLimit += 20
                                    },
                                    modifier = Modifier.fillMaxWidth(),
                                )
                            }
                        }
                    }
                }
                // Clears the fixed selection bar so the last row isn't hidden behind it.
                if (selecting) Spacer(modifier = Modifier.height(96.dp))
            }
        }

        // Bulk-action bar — floats over the bottom while selecting.
        if (selecting) {
            SelectionActionBar(
                modifier = Modifier.align(Alignment.BottomCenter),
                section = section,
                count = selected.size,
                restoreCount = currentPhotos.count { it.id in selected && it.deletionStartedAt == null },
                allSelected = currentPhotos.isNotEmpty() && selected.size == currentPhotos.size,
                busy = busy,
                onToggleAll = {
                    selected = if (selected.size == currentPhotos.size) emptySet() else currentPhotos.map { it.id }.toSet()
                },
                onDelete = {
                    if (busy || selected.isEmpty()) return@SelectionActionBar
                    busy = true
                    val ids = selected.toList()
                    scope.launch {
                        val message = viewModel.trashPhotos(event.id, ids).fold(
                            onSuccess = { n -> "$n photo${if (n > 1) "s" else ""} moved to Deleted." },
                            onFailure = { it.message ?: "Something went wrong." },
                        )
                        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                        exitSelect()
                        busy = false
                    }
                },
                onRestore = {
                    if (busy || selected.isEmpty()) return@SelectionActionBar
                    busy = true
                    val ids = currentPhotos.filter { it.id in selected && it.deletionStartedAt == null }.map { it.id }
                    scope.launch {
                        val message = viewModel.restorePhotos(event.id, ids).fold(
                            onSuccess = { n -> "$n photo${if (n > 1) "s" else ""} restored." },
                            onFailure = { it.message ?: "Something went wrong." },
                        )
                        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                        exitSelect()
                        busy = false
                    }
                },
                onPermanent = { if (!busy && selected.isNotEmpty()) confirmPermanent = true },
                onDone = { exitSelect() },
            )
        }
    }

    // Reuses the runner-side PhotoPreview lightbox in OwnerReview mode — swipe
    // through the current tab's photos, no purchase UI. Disabled in select mode.
    val idx = selectedIndex
    if (!selecting && idx != null && currentPhotos.isNotEmpty()) {
        val previewPhotos = remember(currentPhotos, event.name) {
            currentPhotos.map { it.toOwnerPreviewData(event.name) }
        }
        val safeIdx = idx.coerceIn(0, previewPhotos.size - 1)
        PhotoPreview(
            photos = previewPhotos,
            currentIndex = safeIdx,
            onClose = { selectedIndex = null },
            onIndexChange = { selectedIndex = it },
            // Two hops, because the library listing only carries the watermarked
            // thumbnail: resolve the presigned original, then hand it to the same
            // saver the runner order screens use.
            onDownload = { photo ->
                if (!downloading) {
                    downloading = true
                    scope.launch {
                        val message = viewModel.resolvePhotoDownloadUrl(photo.id).fold(
                            onSuccess = { url ->
                                val bib = currentPhotos.firstOrNull { it.id == photo.id }?.bib
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

    if (confirmPermanent) {
        val n = selected.size
        AlertDialog(
            onDismissRequest = { if (!busy) confirmPermanent = false },
            containerColor = Bone,
            title = {
                Text(
                    "Permanently delete $n photo${if (n > 1) "s" else ""}?",
                    style = Typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Ink,
                )
            },
            text = {
                Text(
                    "This can't be undone — the original files are removed from storage. " +
                        "Photos that have already sold are kept so their buyers can still download them.",
                    style = Typography.bodyMedium,
                    color = InkSoft,
                )
            },
            confirmButton = {
                TextButton(
                    enabled = !busy,
                    onClick = {
                        if (busy) return@TextButton
                        busy = true
                        val ids = selected.toList()
                        scope.launch {
                            val message = viewModel.permanentDeletePhotos(event.id, ids).fold(
                                onSuccess = { r ->
                                    selected = r.failed.toSet()
                                    "${r.deleted.size} permanently deleted." +
                                        (if (r.blockedSold.isNotEmpty()) " ${r.blockedSold.size} kept for buyers." else "") +
                                        (if (r.blockedPending.isNotEmpty()) " ${r.blockedPending.size} kept while payment is pending or being checked." else "") +
                                        (if (r.failed.isNotEmpty()) " ${r.failed.size} couldn't be deleted. Retry Delete forever to finish cleanup." else "")
                                },
                                onFailure = { it.message ?: "Couldn't delete." },
                            )
                            Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                            confirmPermanent = false
                            if (selected.isEmpty()) exitSelect()
                            busy = false
                        }
                    },
                ) { Text("Delete forever", color = ErrorRed, fontWeight = FontWeight.Bold) }
            },
            dismissButton = {
                TextButton(enabled = !busy, onClick = { confirmPermanent = false }) {
                    Text("Cancel", color = Slate)
                }
            },
        )
    }
}

@Composable
private fun LibraryTabs(section: PhotoSection, onSelect: (PhotoSection) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        LibraryTab("Active", section == PhotoSection.Active) { onSelect(PhotoSection.Active) }
        LibraryTab("Deleted", section == PhotoSection.Deleted) { onSelect(PhotoSection.Deleted) }
    }
}

@Composable
private fun LibraryTab(label: String, active: Boolean, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .clip(PillShape)
            .background(if (active) Ink else Bone)
            .border(BorderStroke(1.dp, if (active) Ink else Line), PillShape)
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 9.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = label,
            style = Typography.labelMedium,
            color = if (active) Bone else Slate,
            fontWeight = FontWeight.Bold,
        )
    }
}

// Small outlined action pill for the library header (Select).
@Composable
private fun HeaderPill(text: String, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .clip(PillShape)
            .background(Bone)
            .border(BorderStroke(1.dp, Ink), PillShape)
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 9.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(text, style = Typography.labelMedium, color = Ink, fontWeight = FontWeight.Bold)
    }
}

// Sticky bulk-action bar for select mode. On the Deleted tab it offers Restore
// + Delete forever; on Active, Delete (to trash). No Fresh accent — the Kept
// stat already spends the viewport's one Fresh.
@Composable
private fun SelectionActionBar(
    section: PhotoSection,
    count: Int,
    restoreCount: Int,
    allSelected: Boolean,
    busy: Boolean,
    onToggleAll: () -> Unit,
    onDelete: () -> Unit,
    onRestore: () -> Unit,
    onPermanent: () -> Unit,
    onDone: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(SurfaceWhite)
            .border(BorderStroke(1.dp, Line))
            .navigationBarsPadding()
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Kicker("$count selected")
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = onToggleAll) {
                    Text(if (allSelected) "Clear" else "Select all", color = Ink, style = Typography.labelMedium)
                }
                TextButton(onClick = onDone) {
                    Text("Done", color = Slate, style = Typography.labelMedium)
                }
            }
        }
        if (section == PhotoSection.Active) {
            GhostCta(
                text = if (busy) "Working…" else "Delete $count",
                enabled = !busy && count > 0,
                onClick = onDelete,
                modifier = Modifier.fillMaxWidth(),
            )
        } else {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                GhostCta(
                    text = if (busy) "Working…" else "Restore $restoreCount",
                    enabled = !busy && restoreCount > 0,
                    onClick = onRestore,
                    modifier = Modifier.weight(1f),
                )
                DangerPill(
                    text = "Delete forever",
                    enabled = !busy && count > 0,
                    onClick = onPermanent,
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }
}

// Destructive outlined action — ErrorRed token, PillShape. The confirmation
// AlertDialog carries the real weight; this is the trigger.
@Composable
private fun DangerPill(text: String, enabled: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .height(48.dp)
            .clip(PillShape)
            .background(Bone)
            .border(BorderStroke(1.dp, ErrorRed), PillShape)
            .clickable(enabled = enabled, onClick = onClick)
            .alpha(if (enabled) 1f else 0.5f)
            .padding(horizontal = 16.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(text, style = Typography.labelMedium, color = ErrorRed, fontWeight = FontWeight.Bold, maxLines = 1)
    }
}

@Composable
private fun SharePhotoTile(
    photo: PhotographerLibraryPhotoDto,
    selecting: Boolean,
    selected: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Box(
        modifier = modifier
            .aspectRatio(0.85f)
            .clip(QpCardShape)
            .background(BoneDeep)
            .then(if (selecting && selected) Modifier.border(BorderStroke(2.dp, Ink), QpCardShape) else Modifier)
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
        if (selecting) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(8.dp)
                    .size(28.dp)
                    .clip(PillShape)
                    .background(if (selected) Ink else Bone.copy(alpha = 0.75f))
                    .border(BorderStroke(1.dp, if (selected) Ink else Line), PillShape),
                contentAlignment = Alignment.Center,
            ) {
                if (selected) {
                    Icon(Icons.Default.Check, contentDescription = null, tint = Bone, modifier = Modifier.size(18.dp))
                }
            }
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
private fun ShareChip(label: String, dotColor: Color, enabled: Boolean = true, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .clip(PillShape)
            .background(Bone)
            .border(BorderStroke(1.dp, Line), PillShape)
            .clickable(enabled = enabled) { onClick() }
            .alpha(if (enabled) 1f else 0.5f)
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
