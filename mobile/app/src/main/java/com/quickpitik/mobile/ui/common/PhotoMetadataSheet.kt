package com.quickpitik.mobile.ui.common

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.ModalBottomSheetProperties
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.window.SecureFlagPolicy
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.quickpitik.mobile.data.remote.PhotoMetadataDto
import com.quickpitik.mobile.data.remote.PhotoMetadataFieldsDto
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.Typography
import java.text.NumberFormat
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotoMetadataSheet(
    state: PhotoMetadataState,
    onClose: () -> Unit,
    onRefresh: () -> Unit,
    secure: Boolean,
) {
    if (state == PhotoMetadataState.Closed) return
    ModalBottomSheet(
        onDismissRequest = onClose,
        sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
        containerColor = Bone,
        properties = ModalBottomSheetProperties(
            securePolicy = if (secure) SecureFlagPolicy.SecureOn else SecureFlagPolicy.Inherit,
        ),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .navigationBarsPadding()
                .padding(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text("Photo metadata", style = Typography.titleLarge, color = Ink)
            when (state) {
                is PhotoMetadataState.Loading -> Text("Loading metadata…", style = Typography.bodyMedium, color = Slate)
                is PhotoMetadataState.LocalLoading -> Text("Loading metadata…", style = Typography.bodyMedium, color = Slate)
                is PhotoMetadataState.Pending -> {
                    Text("Metadata is being prepared.", style = Typography.bodyMedium, color = Slate)
                    GhostCta("Refresh", onRefresh, Modifier.fillMaxWidth())
                }
                is PhotoMetadataState.Data -> {
                    metadataDetails(state.metadata.fields, state.metadata.originalFilename, state.metadata.fileSizeBytes)
                }
                is PhotoMetadataState.LocalData -> {
                    Text("From phone original", style = Typography.bodySmall, color = Slate)
                    metadataDetails(state.metadata.fields, state.metadata.originalFilename, state.metadata.fileSizeBytes)
                }
                is PhotoMetadataState.LocalEmpty -> {
                    Text("From phone original", style = Typography.bodySmall, color = Slate)
                    Text("No camera metadata was found.", style = Typography.bodyMedium, color = Slate)
                    fileFacts(state.metadata.originalFilename, state.metadata.fileSizeBytes)
                }
                is PhotoMetadataState.LocalUnavailable -> Text("Metadata is unavailable for this photo.", style = Typography.bodyMedium, color = Slate)
                is PhotoMetadataState.Empty -> {
                    Text("No camera metadata was found.", style = Typography.bodyMedium, color = Slate)
                    fileFacts(state.metadata.originalFilename, state.metadata.fileSizeBytes)
                }
                is PhotoMetadataState.Unavailable -> {
                    Text("Metadata is unavailable for this photo.", style = Typography.bodyMedium, color = Slate)
                    fileFacts(state.metadata.originalFilename, state.metadata.fileSizeBytes)
                }
                PhotoMetadataState.Denied -> Text("Metadata is no longer available", style = Typography.bodyMedium, color = Slate)
                is PhotoMetadataState.Failed -> {
                    Text("Metadata could not be loaded.", style = Typography.bodyMedium, color = Slate)
                    GhostCta("Retry", onRefresh, Modifier.fillMaxWidth())
                }
                PhotoMetadataState.Closed -> Unit
            }
            GhostCta("Close metadata", onClose, Modifier.fillMaxWidth())
        }
    }
}

@Composable
private fun metadataDetails(fields: PhotoMetadataFieldsDto?, filename: String?, size: Long?) {
    metadataSection("Camera") {
        metadataRow("Model", cameraModel(fields))
        metadataRow("Lens", fields?.lens)
    }
    metadataSection("Exposure") {
        metadataRow("Aperture", fields?.aperture?.let { "f/$it" })
        metadataRow("Exposure", fields?.exposureTime)
        metadataRow("ISO", fields?.iso?.toString())
        metadataRow("Flash", fields?.flash)
        metadataRow("Exposure mode", fields?.exposureMode)
        metadataRow("White balance", fields?.whiteBalance)
    }
    metadataSection("Attribution and capture") {
        metadataRow("Author", fields?.author)
        metadataRow("Copyright", fields?.copyright)
        metadataRow("Date", fields?.takenAtLocal?.replaceFirst('T', ' ')?.let {
            it + (fields.takenAtOffset?.let { offset -> " $offset" } ?: "")
        })
    }
    fileFacts(filename, size)
}

private fun cameraModel(fields: PhotoMetadataFieldsDto?): String? {
    val make = fields?.cameraMake?.trim().orEmpty()
    val model = fields?.cameraModel?.trim().orEmpty()
    if (make.isEmpty()) return model
    if (model.isEmpty()) return make
    return if (model.equals(make, true) || model.startsWith("$make ", true)) model else "$make $model"
}

@Composable
private fun metadataSection(title: String, content: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        HorizontalDivider(color = Line)
        Text(title, style = Typography.titleMedium, color = Ink)
        content()
    }
}

@Composable
private fun metadataRow(label: String, value: String?, modifier: Modifier = Modifier) {
    Column(modifier.fillMaxWidth()) {
        Text(label, style = Typography.bodySmall, color = Slate)
        Text(value?.takeIf { it.isNotEmpty() } ?: "Not recorded", style = Typography.bodyMedium, color = Ink)
    }
}

@Composable
private fun fileFacts(filename: String?, size: Long?) {
    metadataSection("File") {
        metadataRow("Name", filename)
        val formatted = size?.let { String.format(Locale.US, "%.1f MiB", it / 1_048_576.0) }
        val exact = size?.let { NumberFormat.getIntegerInstance(Locale.US).format(it) }
        metadataRow("Original size", formatted,
            if (exact == null) Modifier else Modifier.semantics {
                contentDescription = "Original size, $formatted, $exact bytes"
            },
        )
    }
}

@Preview
@Composable
private fun PhotoMetadataSheetPreview() {
    PhotoMetadataSheet(
        PhotoMetadataState.Data(PhotoMetadataDto("photo", "ready", "race.jpg", 1_572_864,
            PhotoMetadataFieldsDto("Canon", "EOS R6", "RF 70-200mm", 2.8, "1/500 s", 400,
                "Ana", null, "2026-09-28T07:45:12", "+08:00", null, "Manual", "Auto"))),
        onClose = {}, onRefresh = {}, secure = false,
    )
}

@Preview
@Composable
private fun PhotoMetadataSheetPendingPreview() {
    PhotoMetadataSheet(PhotoMetadataState.Pending(PhotoMetadataDto("photo", "pending", null, null, null)),
        onClose = {}, onRefresh = {}, secure = false)
}
