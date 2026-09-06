package com.quickpitik.mobile.ui.runner

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.PhotoVerifyResultDto
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorView
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.QuickPitikMobileTheme
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.StatusChip
import com.quickpitik.mobile.ui.theme.StatusTone
import com.quickpitik.mobile.ui.theme.Typography

// Mobile mirror of the website /verify tool ("Whose photo is this?"). Pick an
// image → upload → attribution card. Placement note (product decision): the
// website's /verify is a public route; on mobile it lives behind login in
// Settings (marketing/splash pages were retired) — movable if we want it public.
@Composable
fun VerifyScreen(
    viewModel: VerifyViewModel,
    onBack: () -> Unit,
    onOpenPhotographer: ((String) -> Unit)? = null,
) {
    val state by viewModel.state.collectAsState()
    var pickedUri by remember { mutableStateOf<Uri?>(null) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            pickedUri = uri
            viewModel.reset()
        }
    }
    val uploading = state is VerifyState.Uploading

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Bone)
            .statusBarsPadding()
            .navigationBarsPadding(),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack, modifier = Modifier.size(48.dp)) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Ink)
            }
            Text(
                text = "Whose photo is this?",
                style = Typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = Ink,
            )
        }

        Column(
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp)
                .padding(bottom = 24.dp),
        ) {
            Text(
                text = "Drop in a race photo and we'll tell you which QuickPitik photographer shot it — a screenshot or a crop works too.",
                style = Typography.bodyMedium,
                color = Slate,
            )
            Spacer(Modifier.height(16.dp))

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(4f / 3f)
                    .clip(QpCardShape)
                    .background(BoneDeep)
                    .border(1.dp, Line, QpCardShape)
                    .clickable(enabled = !uploading) { picker.launch("image/*") },
                contentAlignment = Alignment.Center,
            ) {
                val uri = pickedUri
                if (uri != null) {
                    AsyncImage(
                        model = uri,
                        contentDescription = "Chosen photo",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize(),
                    )
                } else {
                    Text("Tap to choose a photo", style = Typography.bodyMedium, color = SlateSoft)
                }
            }

            Spacer(Modifier.height(16.dp))
            PrimaryCta(
                text = "Check this photo",
                onClick = { pickedUri?.let(viewModel::verify) },
                enabled = pickedUri != null && !uploading,
                loading = uploading,
                modifier = Modifier.fillMaxWidth(),
            )

            Spacer(Modifier.height(20.dp))
            when (val s = state) {
                is VerifyState.Idle, is VerifyState.Uploading -> Unit // the CTA carries the spinner
                is VerifyState.Result -> VerifyResultCard(s.result, onOpenPhotographer)
                is VerifyState.Error -> ErrorView(
                    message = s.message,
                    onRetry = { pickedUri?.let(viewModel::verify) },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}

@Composable
private fun VerifyResultCard(result: PhotoVerifyResultDto, onOpenPhotographer: ((String) -> Unit)?) {
    QpCard(modifier = Modifier.fillMaxWidth()) {
        if (result.matched) {
            Kicker("Match found")
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Yes — a QuickPitik photo.",
                style = Typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = Ink,
            )
            Spacer(Modifier.height(12.dp))
            val handle = result.photographerHandle
            if (handle != null && onOpenPhotographer != null) {
                Text(
                    text = "Taken by @$handle",
                    style = Typography.bodyMedium,
                    color = Ink,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier
                        .clickable { onOpenPhotographer(handle) }
                        .padding(vertical = 4.dp),
                )
            } else {
                // Null handle = an unverified photographer: name is plain text,
                // never a tap target (a link would route to /{null}).
                Text(
                    text = "Taken by ${result.photographerName ?: "a QuickPitik photographer"}",
                    style = Typography.bodyMedium,
                    color = Ink,
                )
            }
            if (!result.eventName.isNullOrBlank()) {
                Spacer(Modifier.height(4.dp))
                Text(
                    text = result.eventName + (result.eventDate?.let { " · $it" } ?: ""),
                    style = Typography.bodySmall,
                    color = Slate,
                )
            }
            result.confidence?.let { conf ->
                Spacer(Modifier.height(12.dp))
                StatusChip(
                    text = "$conf match",
                    tone = if (conf.equals("strong", ignoreCase = true)) StatusTone.Approved else StatusTone.Warning,
                )
            }
        } else {
            Kicker("No match")
            Spacer(Modifier.height(8.dp))
            Text(
                text = "We don't recognize this photo.",
                style = Typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = Ink,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "It may not be a QuickPitik photo, or it's been edited past what we can match.",
                style = Typography.bodySmall,
                color = Slate,
            )
        }
    }
}

@Preview
@Composable
private fun VerifyResultMatchedPreview() {
    QuickPitikMobileTheme {
        VerifyResultCard(
            PhotoVerifyResultDto(
                matched = true,
                confidence = "strong",
                photographerName = "Juan Dela Cruz",
                photographerHandle = "juanshots",
                eventName = "Cebu City Marathon",
                eventDate = "2026-09-20",
            ),
            onOpenPhotographer = {},
        )
    }
}

@Preview
@Composable
private fun VerifyResultNoMatchPreview() {
    QuickPitikMobileTheme {
        VerifyResultCard(PhotoVerifyResultDto(matched = false), onOpenPhotographer = null)
    }
}
