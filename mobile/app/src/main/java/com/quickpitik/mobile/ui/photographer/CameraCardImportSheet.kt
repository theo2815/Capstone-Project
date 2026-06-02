package com.quickpitik.mobile.ui.photographer

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.NumeralStyle
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.SuccessGreen
import com.quickpitik.mobile.ui.theme.TileShape
import com.quickpitik.mobile.ui.theme.Typography
import com.quickpitik.mobile.data.usb.ptp.CardPhoto

/**
 * Increment 1 of the manual camera-import flow: enumerate-only browser sheet.
 *
 * Mounted by `TetherConsoleView` whenever `viewModel.cardBrowseState` is not
 * Idle. Renders four states under one ModalBottomSheet — Opening / Scanning /
 * Loaded / Error — and exposes a single Retry path for the Error state. There
 * is no selection or transfer yet; those land in Increments 2 + 3.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraCardImportSheet(
    state: CardBrowseState,
    onDismiss: () -> Unit,
    onRetry: () -> Unit,
    onToggleSelect: (Long) -> Unit,
    onSelectAll: () -> Unit,
    onClearSelection: () -> Unit,
    onImport: () -> Unit,
    onCancelImport: () -> Unit,
    onRetryFailed: () -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Bone,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp)
                .padding(top = 4.dp, bottom = 24.dp)
                .navigationBarsPadding(),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Kicker(text = "Import from card", color = SlateSoft)

            // AnimatedContent so the swap from Scanning → Loaded reads as one
            // surface morphing (Mobile Design skill: AnimatedContent for state
            // swaps within a slot).
            //
            // contentKey by class — crossfade ONLY on type transitions
            // (Loaded ↔ Importing ↔ Error). Without this, every checkbox
            // toggle creates a new Loaded(...) value, AnimatedContent treats
            // it as a state swap, and the entire LoadedView (including its
            // LazyColumn scroll position) gets torn down + rebuilt — the
            // 2026-05-28 "scroll jumps to top when I tap" bug.
            AnimatedContent(
                targetState = state,
                contentKey = { it::class },
                transitionSpec = { fadeIn() togetherWith fadeOut() },
                label = "card-browse-content",
            ) { current ->
                when (current) {
                    CardBrowseState.Idle ->
                        // Defensive: sheet shouldn't be visible while Idle, but
                        // AnimatedContent may briefly render the previous value
                        // during dismissal. Empty Box keeps the animation clean.
                        Box(modifier = Modifier.fillMaxWidth())
                    CardBrowseState.Opening -> OpeningView()
                    is CardBrowseState.Scanning -> ScanningView(current)
                    is CardBrowseState.Loaded -> LoadedView(
                        state = current,
                        onToggleSelect = onToggleSelect,
                        onSelectAll = onSelectAll,
                        onClearSelection = onClearSelection,
                        onImport = onImport,
                    )
                    is CardBrowseState.Importing -> ImportingView(current, onCancelImport)
                    is CardBrowseState.ImportDone -> ImportDoneView(current, onDone = onDismiss, onRetryFailed = onRetryFailed)
                    is CardBrowseState.Error -> ErrorView(current, onRetry)
                }
            }
        }
    }
}

@Composable
private fun OpeningView() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        CircularProgressIndicator(
            color = Slate,
            strokeWidth = 2.dp,
            modifier = Modifier.size(28.dp),
        )
        Text(
            text = "Opening camera session…",
            color = Slate,
            style = Typography.bodyMedium,
        )
    }
}

@Composable
private fun ScanningView(state: CardBrowseState.Scanning) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = "Reading card…",
            color = Ink,
            fontSize = 18.sp,
            fontWeight = FontWeight.SemiBold,
        )
        val fraction = if (state.total > 0) state.seen.toFloat() / state.total.toFloat() else 0f
        LinearProgressIndicator(
            progress = fraction,
            color = Slate,
            trackColor = Line,
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 6.dp)
                .clip(PillShape),
        )
        Text(
            text = "${state.seen} of ${state.total}",
            color = SlateSoft,
            style = NumeralStyle.copy(fontSize = 13.sp),
        )
    }
}

@Composable
private fun LoadedView(
    state: CardBrowseState.Loaded,
    onToggleSelect: (Long) -> Unit,
    onSelectAll: () -> Unit,
    onClearSelection: () -> Unit,
    onImport: () -> Unit,
) {
    val photos = state.photos
    if (photos.isEmpty()) {
        EmptyCardView()
        return
    }
    val selectedCount = state.selectedHandles.size
    val importedCount = state.importedHandles.size
    val importableCount = photos.size - importedCount
    val allImportableSelected = importableCount > 0 && selectedCount == importableCount
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.Bottom,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                text = "${photos.size} photos on card",
                color = Ink,
                fontSize = 18.sp,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "JPEG only",
                color = SlateSoft,
                style = Typography.bodyMedium,
            )
        }
        SelectionHeader(
            selectedCount = selectedCount,
            importedCount = importedCount,
            allImportableSelected = allImportableSelected,
            anyImportable = importableCount > 0,
            onSelectAll = onSelectAll,
            onClearSelection = onClearSelection,
        )
        Divider(color = Line)
        // Cap the LazyColumn so the bottom Import CTA stays visible without
        // pushing the sheet to full-screen height on long card listings.
        // Remembered LazyListState is the belt-and-suspenders for scroll
        // preservation — paired with AnimatedContent.contentKey above, the
        // list never loses position on a checkbox tap. It DOES reset when
        // LoadedView leaves composition (type transition out of Loaded),
        // which is the right behaviour: re-browse should start at the top.
        val listState = rememberLazyListState()
        LazyColumn(
            state = listState,
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(max = 420.dp),
            verticalArrangement = Arrangement.spacedBy(0.dp),
        ) {
            items(photos, key = { it.handle }) { photo ->
                val isImported = photo.handle in state.importedHandles
                CardPhotoRow(
                    photo = photo,
                    selected = photo.handle in state.selectedHandles,
                    imported = isImported,
                    onToggle = if (isImported) null else ({ onToggleSelect(photo.handle) }),
                )
                Divider(color = Line)
            }
        }
        // Footer Fresh CTA — the single Fresh in this sheet (matches one-per-
        // viewport rule; the Capture tab's Simulate CTA is occluded while the
        // sheet is open).
        PrimaryCta(
            text = importLabel(selectedCount),
            onClick = onImport,
            modifier = Modifier.fillMaxWidth(),
            enabled = selectedCount > 0,
        )
    }
}

@Composable
private fun SelectionHeader(
    selectedCount: Int,
    importedCount: Int,
    allImportableSelected: Boolean,
    anyImportable: Boolean,
    onSelectAll: () -> Unit,
    onClearSelection: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 40.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        val counter = buildString {
            append(if (selectedCount == 0) "Nothing selected" else "$selectedCount selected")
            if (importedCount > 0) append(" · $importedCount already imported")
        }
        Text(
            text = counter,
            color = if (selectedCount == 0) SlateSoft else Ink,
            style = Typography.bodyMedium,
            fontWeight = if (selectedCount == 0) FontWeight.Normal else FontWeight.Medium,
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(
                onClick = onSelectAll,
                enabled = anyImportable && !allImportableSelected,
            ) {
                Text(
                    text = "Select all",
                    color = if (anyImportable && !allImportableSelected) Ink else SlateSoft,
                    fontWeight = FontWeight.Medium,
                )
            }
            TextButton(
                onClick = onClearSelection,
                enabled = selectedCount > 0,
            ) {
                Text(
                    text = "Clear",
                    color = if (selectedCount > 0) Ink else SlateSoft,
                    fontWeight = FontWeight.Medium,
                )
            }
        }
    }
}

@Composable
private fun CardPhotoRow(
    photo: CardPhoto,
    selected: Boolean,
    imported: Boolean,
    onToggle: (() -> Unit)?,
) {
    val rowModifier = Modifier
        .fillMaxWidth()
        .heightIn(min = 56.dp)
        .then(
            if (onToggle != null) Modifier.toggleable(
                value = selected,
                onValueChange = { onToggle() },
                role = Role.Checkbox,
            ) else Modifier
        )
        .padding(vertical = 8.dp)
    Row(
        modifier = rowModifier,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Checkbox(
            checked = selected || imported,
            onCheckedChange = null, // the whole row is toggleable; visual only here
            enabled = !imported,
            colors = CheckboxDefaults.colors(
                checkedColor = if (imported) SuccessGreen else Ink,
                uncheckedColor = SlateSoft,
                checkmarkColor = Bone,
                disabledCheckedColor = SuccessGreen,
            ),
        )
        Spacer(modifier = Modifier.width(8.dp))
        // Increment 5 — real thumbnail when the camera returned one for this
        // handle; BoneDeep TileShape fallback otherwise (RAW-only or older
        // bodies that don't expose GetThumb). Decoded once per handle via
        // remember(photo.handle) — the bytes don't change so the Bitmap is
        // reusable across recompositions of the same row.
        val thumb = remember(photo.handle) {
            photo.thumbnailBytes?.let {
                runCatching { BitmapFactory.decodeByteArray(it, 0, it.size) }
                    .getOrNull()
                    ?.asImageBitmap()
            }
        }
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(TileShape)
                .background(BoneDeep),
        ) {
            if (thumb != null) {
                Image(
                    bitmap = thumb,
                    contentDescription = null,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
            }
        }
        Spacer(modifier = Modifier.width(14.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = photo.filename,
                color = if (imported) SlateSoft else Ink,
                fontSize = 15.sp,
                fontWeight = FontWeight.Medium,
            )
            Spacer(modifier = Modifier.heightIn(min = 2.dp))
            Text(
                text = if (imported) "Imported · ${formatBytes(photo.sizeBytes)}" else formatBytes(photo.sizeBytes),
                color = if (imported) SuccessGreen else SlateSoft,
                style = NumeralStyle.copy(fontSize = 12.sp),
            )
        }
    }
}

/** "Import 1 photo" / "Import 24 photos" / "Select photos to import". */
private fun importLabel(selectedCount: Int): String = when (selectedCount) {
    0 -> "Select photos to import"
    1 -> "Import 1 photo"
    else -> "Import $selectedCount photos"
}

@Composable
private fun EmptyCardView() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 28.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(
            text = "No JPEGs on the card",
            color = Ink,
            fontSize = 18.sp,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Set the camera to JPEG (or JPEG+RAW) and shoot a frame, then re-open this sheet.",
            color = Slate,
            style = Typography.bodyMedium,
        )
    }
}

@Composable
private fun ImportingView(state: CardBrowseState.Importing, onCancel: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = "Importing…",
            color = Ink,
            fontSize = 18.sp,
            fontWeight = FontWeight.SemiBold,
        )
        val fraction = if (state.total > 0) state.seen.toFloat() / state.total.toFloat() else 0f
        LinearProgressIndicator(
            progress = fraction,
            color = Slate,
            trackColor = Line,
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 6.dp)
                .clip(PillShape),
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                text = "${state.seen} of ${state.total}",
                color = SlateSoft,
                style = NumeralStyle.copy(fontSize = 13.sp),
            )
            if (state.failed > 0) {
                Text(
                    text = "${state.failed} failed",
                    color = ErrorRed,
                    style = NumeralStyle.copy(fontSize = 13.sp),
                )
            }
        }
        Text(
            text = "Stay on this screen until the import finishes. Uploads continue in the background once they're queued.",
            color = Slate,
            style = Typography.bodyMedium,
            lineHeight = 18.sp,
        )
        GhostCta(
            text = "Cancel import",
            onClick = onCancel,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun ImportDoneView(
    state: CardBrowseState.ImportDone,
    onDone: () -> Unit,
    onRetryFailed: () -> Unit,
) {
    val succeededCount = state.succeededHandles.size
    val failedCount = state.failedHandles.size
    val total = succeededCount + failedCount
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = when {
                succeededCount == 0 -> "Import failed"
                failedCount == 0 -> "All imported"
                else -> "Partial import"
            },
            color = Ink,
            fontSize = 18.sp,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = importDoneBody(succeededCount, failedCount, total),
            color = Slate,
            style = Typography.bodyMedium,
            lineHeight = 20.sp,
        )
        if (failedCount > 0) {
            GhostCta(
                text = "Retry ${failedCount} failed",
                onClick = onRetryFailed,
                modifier = Modifier.fillMaxWidth(),
            )
        }
        GhostCta(
            text = "Done",
            onClick = onDone,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

private fun importDoneBody(succeeded: Int, failed: Int, total: Int): String = when {
    succeeded == 0 -> "No photos made it to the queue. Check the cable and the camera's USB mode, then try again."
    failed == 0 -> "$succeeded ${pluralPhotos(succeeded)} queued. Track the upload in the sync queue on the Capture tab."
    else -> "$succeeded of $total ${pluralPhotos(total)} queued. Tap Retry to re-pull the $failed that failed."
}

private fun pluralPhotos(n: Int): String = if (n == 1) "photo" else "photos"

@Composable
private fun ErrorView(state: CardBrowseState.Error, onRetry: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Kicker(text = "Couldn't read card", color = ErrorRed)
        Text(
            text = state.message,
            color = Slate,
            style = Typography.bodyMedium,
        )
        // Always re-browse on retry — works for browse-time AND import-time
        // errors. For import-time, the dedupe markers on the fresh Loaded view
        // preserve "what already succeeded" context (Increment 4 D), so the
        // photographer can re-select just what's left without re-uploading.
        GhostCta(
            text = "Browse card again",
            onClick = onRetry,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

/** Bytes → "412 KB" / "3.2 MB". One decimal once we cross the megabyte line. */
private fun formatBytes(bytes: Long): String {
    if (bytes < 1024) return "$bytes B"
    val kb = bytes / 1024.0
    if (kb < 1024) return "${kb.toInt()} KB"
    val mb = kb / 1024.0
    return "%.1f MB".format(mb)
}
