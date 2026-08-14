package com.quickpitik.mobile.ui.photographer

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Divider
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.PayoutAccountDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.SocialLinkDto
import com.quickpitik.mobile.ui.theme.BadgeShape
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.ErrorView
import com.quickpitik.mobile.ui.theme.FieldShape
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.LoadingSkeleton
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.StatusChip
import com.quickpitik.mobile.ui.theme.StatusTone
import com.quickpitik.mobile.ui.theme.Typography
import com.quickpitik.mobile.ui.theme.WarningOrange

// ─── Reference data ──────────────────────────────────────────────────────────
// PH regions/provinces is pre-existing inline data (separate cleanup task in
// mobile/tasks.md "Now" — backend-owned reference data). Keeping the structure
// surgical to today's redesign.

data class PHProvince(val code: String, val name: String)
data class PHRegion(val code: String, val name: String, val provinces: List<PHProvince>)

val PH_REGIONS = listOf(
    PHRegion("ncr", "National Capital Region (NCR)", listOf(PHProvince("metro-manila", "Metro Manila"))),
    PHRegion("car", "Cordillera Administrative Region (CAR)", listOf(
        PHProvince("abra", "Abra"), PHProvince("apayao", "Apayao"), PHProvince("benguet", "Benguet"),
        PHProvince("ifugao", "Ifugao"), PHProvince("kalinga", "Kalinga"), PHProvince("mountain-province", "Mountain Province"),
    )),
    PHRegion("region-1", "Region I (Ilocos Region)", listOf(
        PHProvince("ilocos-norte", "Ilocos Norte"), PHProvince("ilocos-sur", "Ilocos Sur"),
        PHProvince("la-union", "La Union"), PHProvince("pangasinan", "Pangasinan"),
    )),
    PHRegion("region-2", "Region II (Cagayan Valley)", listOf(
        PHProvince("batanes", "Batanes"), PHProvince("cagayan", "Cagayan"),
        PHProvince("isabela", "Isabela"), PHProvince("nueva-vizcaya", "Nueva Vizcaya"), PHProvince("quirino", "Quirino"),
    )),
    PHRegion("region-3", "Region III (Central Luzon)", listOf(
        PHProvince("aurora", "Aurora"), PHProvince("bataan", "Bataan"), PHProvince("bulacan", "Bulacan"),
        PHProvince("nueva-ecija", "Nueva Ecija"), PHProvince("pampanga", "Pampanga"),
        PHProvince("tarlac", "Tarlac"), PHProvince("zambales", "Zambales"),
    )),
    PHRegion("region-4a", "Region IV-A (CALABARZON)", listOf(
        PHProvince("batangas", "Batangas"), PHProvince("cavite", "Cavite"),
        PHProvince("laguna", "Laguna"), PHProvince("quezon", "Quezon"), PHProvince("rizal", "Rizal"),
    )),
    PHRegion("region-4b", "Region IV-B (MIMAROPA)", listOf(
        PHProvince("marinduque", "Marinduque"), PHProvince("occidental-mindoro", "Occidental Mindoro"),
        PHProvince("oriental-mindoro", "Oriental Mindoro"), PHProvince("palawan", "Palawan"), PHProvince("romblon", "Romblon"),
    )),
    PHRegion("region-5", "Region V (Bicol Region)", listOf(
        PHProvince("albay", "Albay"), PHProvince("camarines-norte", "Camarines Norte"),
        PHProvince("camarines-sur", "Camarines Sur"), PHProvince("catanduanes", "Catanduanes"),
        PHProvince("masbate", "Masbate"), PHProvince("sorsogon", "Sorsogon"),
    )),
    PHRegion("region-6", "Region VI (Western Visayas)", listOf(
        PHProvince("aklan", "Aklan"), PHProvince("antique", "Antique"), PHProvince("capiz", "Capiz"),
        PHProvince("guimaras", "Guimaras"), PHProvince("iloilo", "Iloilo"), PHProvince("negros-occidental", "Negros Occidental"),
    )),
    PHRegion("region-7", "Region VII (Central Visayas)", listOf(
        PHProvince("bohol", "Bohol"), PHProvince("cebu", "Cebu"),
        PHProvince("negros-oriental", "Negros Oriental"), PHProvince("siquijor", "Siquijor"),
    )),
    PHRegion("region-8", "Region VIII (Eastern Visayas)", listOf(
        PHProvince("biliran", "Biliran"), PHProvince("leyte", "Leyte"), PHProvince("northern-samar", "Northern Samar"),
        PHProvince("samar", "Samar"), PHProvince("southern-leyte", "Southern Leyte"), PHProvince("eastern-samar", "Eastern Samar"),
    )),
    PHRegion("region-9", "Region IX (Zamboanga Peninsula)", listOf(
        PHProvince("zamboanga-del-norte", "Zamboanga del Norte"),
        PHProvince("zamboanga-del-sur", "Zamboanga del Sur"),
        PHProvince("zamboanga-sibugay", "Zamboanga Sibugay"),
    )),
    PHRegion("region-10", "Region X (Northern Mindanao)", listOf(
        PHProvince("bukidnon", "Bukidnon"), PHProvince("camiguin", "Camiguin"),
        PHProvince("lanao-del-norte", "Lanao del Norte"), PHProvince("misamis-occidental", "Misamis Occidental"),
        PHProvince("misamis-oriental", "Misamis Oriental"),
    )),
    PHRegion("region-11", "Region XI (Davao Region)", listOf(
        PHProvince("davao-de-oro", "Davao de Oro"), PHProvince("davao-del-norte", "Davao del Norte"),
        PHProvince("davao-del-sur", "Davao del Sur"), PHProvince("davao-occidental", "Davao Occidental"),
        PHProvince("davao-oriental", "Davao Oriental"),
    )),
    PHRegion("region-12", "Region XII (SOCCSKSARGEN)", listOf(
        PHProvince("cotabato", "Cotabato"), PHProvince("sarangani", "Sarangani"),
        PHProvince("south-cotabato", "South Cotabato"), PHProvince("sultan-kudarat", "Sultan Kudarat"),
    )),
    PHRegion("region-13", "Region XIII (Caraga)", listOf(
        PHProvince("agusan-del-norte", "Agusan del Norte"), PHProvince("agusan-del-sur", "Agusan del Sur"),
        PHProvince("dinagat-islands", "Dinagat Islands"), PHProvince("surigao-del-norte", "Surigao del Norte"),
        PHProvince("surigao-del-sur", "Surigao del Sur"),
    )),
    PHRegion("barmm", "Bangsamoro Autonomous Region in Muslim Mindanao (BARMM)", listOf(
        PHProvince("basilan", "Basilan"), PHProvince("lanao-del-sur", "Lanao del Sur"),
        PHProvince("maguindanao-del-norte", "Maguindanao del Norte"), PHProvince("maguindanao-del-sur", "Maguindanao del Sur"),
        PHProvince("sulu", "Sulu"), PHProvince("tawi-tawi", "Tawi-Tawi"),
    )),
)

// ─── Platform + payout constants (website parity) ────────────────────────────

private val SOCIAL_PLATFORMS = listOf("facebook", "instagram", "tiktok", "x", "youtube", "website")

private val SOCIAL_PLATFORM_LABEL = mapOf(
    "facebook" to "Facebook",
    "instagram" to "Instagram",
    "tiktok" to "TikTok",
    "x" to "X",
    "youtube" to "YouTube",
    "website" to "Website",
)

private val SOCIAL_PLATFORM_TILE = mapOf(
    "facebook" to "FB",
    "instagram" to "IG",
    "tiktok" to "TT",
    "x" to "X",
    "youtube" to "YT",
    "website" to "WEB",
)

private val SOCIAL_URL_PATTERN = mapOf(
    "facebook" to Regex("^https?://(?:www\\.|m\\.|web\\.)?(?:facebook\\.com|fb\\.com|fb\\.me)/", RegexOption.IGNORE_CASE),
    "instagram" to Regex("^https?://(?:www\\.)?instagram\\.com/", RegexOption.IGNORE_CASE),
    "tiktok" to Regex("^https?://(?:www\\.|vm\\.)?tiktok\\.com/", RegexOption.IGNORE_CASE),
    "x" to Regex("^https?://(?:www\\.)?(?:x\\.com|twitter\\.com)/", RegexOption.IGNORE_CASE),
    "youtube" to Regex("^https?://(?:www\\.|m\\.)?(?:youtube\\.com|youtu\\.be)/", RegexOption.IGNORE_CASE),
    "website" to Regex("^https?://[^\\s/]+\\.[^\\s/]+", RegexOption.IGNORE_CASE),
)

private fun validateSocialUrl(platform: String, url: String): String? {
    val trimmed = url.trim()
    if (trimmed.isEmpty()) return "Add a URL or remove this row."
    if (trimmed.length > 2048) return "URL is at most 2048 characters."
    val pattern = SOCIAL_URL_PATTERN[platform] ?: return null
    if (!pattern.containsMatchIn(trimmed)) {
        return if (platform == "website") {
            "URL must start with http:// or https://."
        } else {
            "This doesn't look like a ${SOCIAL_PLATFORM_LABEL[platform] ?: platform} URL."
        }
    }
    return null
}

private val PAYOUT_METHODS = listOf("gcash", "maya", "gotyme")
private val PAYOUT_METHOD_LABEL = mapOf(
    "gcash" to "GCash",
    "maya" to "Maya",
    "gotyme" to "GoTyme Bank",
)

private fun formatPayoutNumber(method: String, raw: String): String {
    val digits = raw.filter { it.isDigit() }
    if (method == "gotyme") {
        val trimmed = digits.take(16)
        return trimmed.chunked(4).joinToString(" ").trim()
    }
    val trimmed = digits.take(11)
    return when {
        trimmed.length <= 4 -> trimmed
        trimmed.length <= 7 -> "${trimmed.substring(0, 4)} ${trimmed.substring(4)}"
        else -> "${trimmed.substring(0, 4)} ${trimmed.substring(4, 7)} ${trimmed.substring(7)}"
    }
}

private fun validatePayoutNumber(method: String, raw: String): String? {
    val d = raw.filter { it.isDigit() }
    if (method == "gcash" || method == "maya") {
        if (d.isEmpty()) return null
        if (d.length > 11) return "Mobile number is at most 11 digits."
        if (d.length != 11) return "Mobile number is 11 digits."
        if (!d.startsWith("09")) return "PH mobile numbers start with 09."
        return null
    }
    if (d.isEmpty()) return null
    if (d.length > 16) return "Account number is at most 16 digits."
    if (d.length < 10) return "Account number is at least 10 digits."
    return null
}

private fun resolveImageUrl(url: String?): String? {
    if (url == null || url.trim().isEmpty()) return null
    // M-2 (2026-05-27 PM): host derived from RetrofitClient.BASE_URL.
    if (url.startsWith("/")) return "${RetrofitClient.backendOrigin}$url"
    return url.replace("localhost", RetrofitClient.backendHost)
        .replace("127.0.0.1", RetrofitClient.backendHost)
}

// ─── Sheet modes ─────────────────────────────────────────────────────────────

private sealed class SocialSheetMode {
    object Add : SocialSheetMode()
    data class Edit(val link: SocialLinkDto) : SocialSheetMode()
}

private sealed class PayoutSheetMode {
    object Add : PayoutSheetMode()
    data class Edit(val account: PayoutAccountDto) : PayoutSheetMode()
}

// ─── Main screen ─────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotographerSettingsScreen(
    viewModel: PhotographerDashboardViewModel,
    onLogout: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val verificationState by viewModel.verificationState.collectAsState()
    val brandSettings by viewModel.brandSettings.collectAsState()
    val payoutAccounts by viewModel.payoutAccounts.collectAsState()
    val socials by viewModel.socials.collectAsState()
    val isSaving by viewModel.isSavingSettings.collectAsState()
    val actionMessage by viewModel.settingsActionState.collectAsState()

    // Batched form inputs (Save button)
    var brandName by remember { mutableStateOf("") }
    var bio by remember { mutableStateOf("") }
    var handle by remember { mutableStateOf("") }
    var regionCode by remember { mutableStateOf("") }
    var provinceCode by remember { mutableStateOf("") }

    var avatarUri by remember { mutableStateOf<String?>(null) }
    var coverUri by remember { mutableStateOf<String?>(null) }
    var watermarkUri by remember { mutableStateOf<String?>(null) }

    var avatarBytes by remember { mutableStateOf<ByteArray?>(null) }
    var coverBytes by remember { mutableStateOf<ByteArray?>(null) }
    var watermarkBytes by remember { mutableStateOf<ByteArray?>(null) }

    val avatarPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
    ) { uri ->
        uri?.let {
            try {
                context.contentResolver.openInputStream(it)?.use { stream ->
                    avatarBytes = stream.readBytes()
                    avatarUri = it.toString()
                }
            } catch (e: Exception) {
                Toast.makeText(context, "Failed to load image", Toast.LENGTH_SHORT).show()
            }
        }
    }
    val coverPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
    ) { uri ->
        uri?.let {
            try {
                context.contentResolver.openInputStream(it)?.use { stream ->
                    coverBytes = stream.readBytes()
                    coverUri = it.toString()
                }
            } catch (e: Exception) {
                Toast.makeText(context, "Failed to load image", Toast.LENGTH_SHORT).show()
            }
        }
    }
    val watermarkPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
    ) { uri ->
        uri?.let {
            try {
                context.contentResolver.openInputStream(it)?.use { stream ->
                    watermarkBytes = stream.readBytes()
                    watermarkUri = it.toString()
                }
            } catch (e: Exception) {
                Toast.makeText(context, "Failed to load image", Toast.LENGTH_SHORT).show()
            }
        }
    }

    LaunchedEffect(actionMessage) {
        actionMessage?.let {
            Toast.makeText(context, it, Toast.LENGTH_LONG).show()
            if (it.startsWith("Success")) {
                avatarUri = null
                coverUri = null
                watermarkUri = null
                avatarBytes = null
                coverBytes = null
                watermarkBytes = null
            }
            viewModel.clearSettingsActionState()
        }
    }

    LaunchedEffect(brandSettings) {
        brandSettings?.let {
            if (brandName.isEmpty()) brandName = it.brandName.orEmpty()
            if (bio.isEmpty()) bio = it.bio.orEmpty()
            if (handle.isEmpty()) handle = it.handle.orEmpty()
            if (regionCode.isEmpty()) regionCode = it.regionCode.orEmpty()
            if (provinceCode.isEmpty()) provinceCode = it.provinceCode.orEmpty()
        }
    }

    var socialSheetMode by remember { mutableStateOf<SocialSheetMode?>(null) }
    var payoutSheetMode by remember { mutableStateOf<PayoutSheetMode?>(null) }
    var socialRowToDelete by remember { mutableStateOf<SocialLinkDto?>(null) }
    var payoutRowToDelete by remember { mutableStateOf<PayoutAccountDto?>(null) }

    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .background(Bone),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp),
    ) {
        item("header") {
            Kicker(text = "Settings · brand + payouts", color = Slate)
        }
        item("verification") {
            VerificationSlab(
                state = verificationState,
                onSubmit = { viewModel.submitVerification() },
                onWithdraw = { viewModel.withdrawVerification() },
            )
        }
        item("publicProfile") {
            HairlineDivider()
            PublicProfileSlab(
                coverRemoteUrl = resolveImageUrl(brandSettings?.coverUrl),
                coverStagedUri = coverUri,
                avatarRemoteUrl = resolveImageUrl(brandSettings?.avatarUrl),
                avatarStagedUri = avatarUri,
                brandName = brandName,
                onBrandNameChange = { brandName = it },
                bio = bio,
                onBioChange = { bio = it },
                onPickAvatar = {
                    avatarPickerLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                    )
                },
                onPickCover = {
                    coverPickerLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                    )
                },
            )
        }
        item("watermark") {
            HairlineDivider()
            WatermarkSlab(
                remoteUrl = resolveImageUrl(brandSettings?.watermarkUrl),
                stagedUri = watermarkUri,
                onPick = {
                    watermarkPickerLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                    )
                },
            )
        }
        item("handle") {
            HairlineDivider()
            HandleSlab(value = handle, onChange = { handle = it })
        }
        item("region") {
            HairlineDivider()
            RegionSlab(
                regionCode = regionCode,
                provinceCode = provinceCode,
                onRegionChange = {
                    regionCode = it
                    provinceCode = ""
                },
                onProvinceChange = { provinceCode = it },
            )
        }
        item("socials") {
            HairlineDivider()
            SocialsSlab(
                socials = socials,
                onAdd = { socialSheetMode = SocialSheetMode.Add },
                onEdit = { socialSheetMode = SocialSheetMode.Edit(it) },
                onAskDelete = { socialRowToDelete = it },
            )
        }
        item("payouts") {
            HairlineDivider()
            PayoutsSlab(
                payouts = payoutAccounts,
                onAdd = { payoutSheetMode = PayoutSheetMode.Add },
                onEdit = { payoutSheetMode = PayoutSheetMode.Edit(it) },
                onAskDelete = { payoutRowToDelete = it },
                onSetPrimary = { viewModel.setPrimaryPayoutAccount(it.id) },
            )
        }
        item("actions") {
            HairlineDivider()
            BottomActions(
                isSaving = isSaving,
                onSave = {
                    viewModel.saveSettings(
                        brandName = brandName,
                        bio = bio,
                        gcashName = "",
                        gcashNumber = "",
                        handle = handle,
                        regionCode = regionCode,
                        provinceCode = provinceCode,
                        socialUrl = "",
                        avatarBytes = avatarBytes,
                        coverBytes = coverBytes,
                        watermarkBytes = watermarkBytes,
                    )
                },
                onSignOut = onLogout,
            )
        }
    }

    socialSheetMode?.let { mode ->
        SocialEditorSheet(
            mode = mode,
            existingSocials = socials,
            onDismiss = { socialSheetMode = null },
            onConfirm = { platform, url ->
                when (mode) {
                    is SocialSheetMode.Add -> viewModel.addSocialAccount(platform, url)
                    is SocialSheetMode.Edit -> viewModel.updateSocialAccount(mode.link.id, url)
                }
                socialSheetMode = null
            },
        )
    }

    payoutSheetMode?.let { mode ->
        PayoutEditorSheet(
            mode = mode,
            onDismiss = { payoutSheetMode = null },
            onConfirm = { method, accountName, accountNumber, qrBytes ->
                when (mode) {
                    is PayoutSheetMode.Add -> viewModel.addPayoutAccount(method, accountName, accountNumber, qrBytes)
                    is PayoutSheetMode.Edit -> {
                        viewModel.updatePayoutAccount(mode.account.id, accountName, accountNumber)
                        if (qrBytes != null) viewModel.uploadPayoutAccountQr(mode.account.id, qrBytes)
                    }
                }
                payoutSheetMode = null
            },
        )
    }

    socialRowToDelete?.let { link ->
        AlertDialog(
            onDismissRequest = { socialRowToDelete = null },
            containerColor = Bone,
            title = { Text("Remove ${SOCIAL_PLATFORM_LABEL[link.platform] ?: link.platform}?", color = Ink) },
            text = { Text("This will remove the link from your public profile.", color = Slate, style = Typography.bodyMedium) },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deleteSocialAccount(link.id)
                    socialRowToDelete = null
                }) { Text("REMOVE", color = ErrorRed, fontWeight = FontWeight.SemiBold) }
            },
            dismissButton = {
                TextButton(onClick = { socialRowToDelete = null }) {
                    Text("CANCEL", color = Slate, fontWeight = FontWeight.SemiBold)
                }
            },
        )
    }

    payoutRowToDelete?.let { account ->
        AlertDialog(
            onDismissRequest = { payoutRowToDelete = null },
            containerColor = Bone,
            title = { Text("Remove this payout?", color = Ink) },
            text = {
                Text(
                    text = "${PAYOUT_METHOD_LABEL[account.method] ?: account.method} · ${account.accountName}",
                    color = Slate,
                    style = Typography.bodyMedium,
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deletePayoutAccount(account.id)
                    payoutRowToDelete = null
                }) { Text("REMOVE", color = ErrorRed, fontWeight = FontWeight.SemiBold) }
            },
            dismissButton = {
                TextButton(onClick = { payoutRowToDelete = null }) {
                    Text("CANCEL", color = Slate, fontWeight = FontWeight.SemiBold)
                }
            },
        )
    }
}

// ─── Hairline divider ────────────────────────────────────────────────────────

@Composable
private fun HairlineDivider() {
    Divider(thickness = 1.dp, color = Line)
}

// ─── Slab: Verification ──────────────────────────────────────────────────────

@Composable
private fun VerificationSlab(
    state: VerificationUiState,
    onSubmit: () -> Unit,
    onWithdraw: () -> Unit,
) {
    Column {
        Kicker(text = "Verification", color = SlateSoft)
        Spacer(modifier = Modifier.height(12.dp))
        when (state) {
            is VerificationUiState.Loading -> {
                LoadingSkeleton(modifier = Modifier.fillMaxWidth().height(40.dp))
            }
            is VerificationUiState.Error -> ErrorView(message = state.message)
            is VerificationUiState.Success -> {
                val data = state.verification
                val status = data.status.lowercase()
                val (chipText, tone) = when (status) {
                    "approved" -> "Approved" to StatusTone.Approved
                    "pending" -> "Under review" to StatusTone.Warning
                    "rejected" -> "Changes requested" to StatusTone.Danger
                    else -> "Incomplete" to StatusTone.Neutral
                }
                StatusChip(text = chipText, tone = tone)
                Spacer(modifier = Modifier.height(10.dp))
                val body = when (status) {
                    "approved" -> "Your profile is public and discoverable by runners."
                    "pending" -> "Sit tight — an admin will review within 1–2 days."
                    "rejected" -> data.suspensionReason ?: "Update your profile and resubmit."
                    else -> "Fill the slabs below, then submit for review."
                }
                Text(text = body, color = Slate, style = Typography.bodyMedium)

                if (!data.missing.isNullOrEmpty() && status != "approved") {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Missing: ${data.missing.joinToString(" · ")}",
                        color = SlateSoft,
                        style = Typography.bodySmall,
                    )
                }

                Spacer(modifier = Modifier.height(14.dp))
                when (status) {
                    "pending" -> GhostCta(
                        text = "Withdraw review",
                        onClick = onWithdraw,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    "approved" -> Unit
                    else -> GhostCta(
                        text = "Submit for review",
                        onClick = onSubmit,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
        }
    }
}

// ─── Slab: Public Profile ────────────────────────────────────────────────────

@Composable
private fun PublicProfileSlab(
    coverRemoteUrl: String?,
    coverStagedUri: String?,
    avatarRemoteUrl: String?,
    avatarStagedUri: String?,
    brandName: String,
    onBrandNameChange: (String) -> Unit,
    bio: String,
    onBioChange: (String) -> Unit,
    onPickAvatar: () -> Unit,
    onPickCover: () -> Unit,
) {
    Column {
        Kicker(text = "Public profile", color = SlateSoft)
        Spacer(modifier = Modifier.height(12.dp))

        // Cover banner — tap to replace
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(16f / 9f)
                .clip(QpCardShape)
                .background(Line)
                .clickable { onPickCover() },
            contentAlignment = Alignment.Center,
        ) {
            val coverModel = coverStagedUri ?: coverRemoteUrl
            if (coverModel != null) {
                AsyncImage(
                    model = coverModel,
                    contentDescription = "Cover banner",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
            } else {
                Text(text = "TAP TO ADD COVER", color = SlateSoft, style = Typography.labelMedium)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Avatar disc + brand name row
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(64.dp)
                    .clip(CircleShape)
                    .background(BoneDeep)
                    .clickable { onPickAvatar() },
                contentAlignment = Alignment.Center,
            ) {
                val avatarModel = avatarStagedUri ?: avatarRemoteUrl
                if (avatarModel != null) {
                    AsyncImage(
                        model = avatarModel,
                        contentDescription = "Avatar",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize(),
                    )
                } else {
                    Text(text = "+", color = Slate, fontSize = 24.sp, fontWeight = FontWeight.SemiBold)
                }
            }
            Spacer(modifier = Modifier.width(12.dp))
            Column {
                Text(
                    text = "Tap to replace",
                    color = Ink,
                    style = Typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                )
                Text(
                    text = "Square JPEG / PNG · ≤ 5 MB",
                    color = SlateSoft,
                    style = Typography.bodySmall,
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        QpFormField(
            label = "Brand name",
            value = brandName,
            onChange = onBrandNameChange,
            placeholder = "e.g., Studio Chan",
        )

        Spacer(modifier = Modifier.height(16.dp))

        QpFormField(
            label = "Bio",
            value = bio,
            onChange = onBioChange,
            placeholder = "A short line runners see on your profile.",
            singleLine = false,
            minLines = 3,
        )
    }
}

// ─── Slab: Watermark ─────────────────────────────────────────────────────────

@Composable
private fun WatermarkSlab(
    remoteUrl: String?,
    stagedUri: String?,
    onPick: () -> Unit,
) {
    Column {
        Kicker(text = "Watermark", color = SlateSoft)
        Spacer(modifier = Modifier.height(12.dp))
        QpCard(modifier = Modifier.fillMaxWidth(), padding = 12.dp) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(72.dp)
                        .clip(QpCardShape)
                        .background(Bone),
                    contentAlignment = Alignment.Center,
                ) {
                    val model = stagedUri ?: remoteUrl
                    if (model != null) {
                        AsyncImage(
                            model = model,
                            contentDescription = "Watermark",
                            contentScale = ContentScale.Fit,
                            modifier = Modifier.fillMaxSize().padding(8.dp),
                        )
                    } else {
                        Text(text = "WM", color = SlateSoft, style = Typography.labelMedium)
                    }
                }
                Spacer(modifier = Modifier.width(14.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (stagedUri != null || remoteUrl != null) "Replace watermark" else "Add a transparent PNG",
                        color = Ink,
                        style = Typography.bodyMedium,
                        fontWeight = FontWeight.Medium,
                    )
                    Text(
                        text = "Burned into every gallery preview.",
                        color = SlateSoft,
                        style = Typography.bodySmall,
                    )
                }
                TextButton(onClick = onPick) {
                    Text(
                        text = if (stagedUri != null || remoteUrl != null) "REPLACE" else "PICK",
                        color = Ink,
                        style = Typography.labelMedium,
                    )
                }
            }
        }
    }
}

// ─── Slab: Handle ────────────────────────────────────────────────────────────

@Composable
private fun HandleSlab(value: String, onChange: (String) -> Unit) {
    Column {
        Kicker(text = "Public handle", color = SlateSoft)
        Spacer(modifier = Modifier.height(12.dp))
        QpFormField(
            label = "@your-handle",
            value = value,
            onChange = onChange,
            placeholder = "studio-chan",
            prefix = "@",
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = "quickpitik.com/@$value".ifBlank { "Your public profile URL appears here." },
            color = SlateSoft,
            style = Typography.bodySmall,
        )
    }
}

// ─── Slab: Region + province ─────────────────────────────────────────────────

@Composable
private fun RegionSlab(
    regionCode: String,
    provinceCode: String,
    onRegionChange: (String) -> Unit,
    onProvinceChange: (String) -> Unit,
) {
    val selectedRegion = PH_REGIONS.firstOrNull { it.code == regionCode }
    val selectedProvince = selectedRegion?.provinces?.firstOrNull { it.code == provinceCode }
    var regionMenuExpanded by remember { mutableStateOf(false) }
    var provinceMenuExpanded by remember { mutableStateOf(false) }

    Column {
        Kicker(text = "Region", color = SlateSoft)
        Spacer(modifier = Modifier.height(12.dp))

        DropdownField(
            label = "Region",
            valueLabel = selectedRegion?.name.orEmpty(),
            placeholder = "Pick a region",
            expanded = regionMenuExpanded,
            onExpand = { regionMenuExpanded = true },
            onCollapse = { regionMenuExpanded = false },
        ) {
            PH_REGIONS.forEach { region ->
                DropdownMenuItem(
                    text = { Text(region.name, color = Ink, style = Typography.bodyMedium) },
                    onClick = {
                        onRegionChange(region.code)
                        regionMenuExpanded = false
                    },
                )
            }
        }

        if (selectedRegion != null) {
            Spacer(modifier = Modifier.height(12.dp))
            DropdownField(
                label = "Province",
                valueLabel = selectedProvince?.name.orEmpty(),
                placeholder = "Pick a province",
                expanded = provinceMenuExpanded,
                onExpand = { provinceMenuExpanded = true },
                onCollapse = { provinceMenuExpanded = false },
            ) {
                selectedRegion.provinces.forEach { province ->
                    DropdownMenuItem(
                        text = { Text(province.name, color = Ink, style = Typography.bodyMedium) },
                        onClick = {
                            onProvinceChange(province.code)
                            provinceMenuExpanded = false
                        },
                    )
                }
            }
        }
    }
}

// ─── Slab: Socials ───────────────────────────────────────────────────────────

@Composable
private fun SocialsSlab(
    socials: List<SocialLinkDto>,
    onAdd: () -> Unit,
    onEdit: (SocialLinkDto) -> Unit,
    onAskDelete: (SocialLinkDto) -> Unit,
) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Kicker(text = "Socials · ${socials.size} of ${SOCIAL_PLATFORMS.size}", color = SlateSoft)
        }
        Spacer(modifier = Modifier.height(12.dp))
        if (socials.isEmpty()) {
            Text(
                text = "No socials added yet. Runners see these on your public profile.",
                color = SlateSoft,
                style = Typography.bodyMedium,
            )
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                socials.forEach { link ->
                    SocialRow(
                        link = link,
                        onEdit = { onEdit(link) },
                        onDelete = { onAskDelete(link) },
                    )
                }
            }
        }
        Spacer(modifier = Modifier.height(12.dp))
        GhostCta(
            text = if (socials.size >= SOCIAL_PLATFORMS.size) "All platforms added" else "Add social link",
            onClick = onAdd,
            enabled = socials.size < SOCIAL_PLATFORMS.size,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun SocialRow(
    link: SocialLinkDto,
    onEdit: () -> Unit,
    onDelete: () -> Unit,
) {
    QpCard(modifier = Modifier.fillMaxWidth(), padding = 12.dp) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(CircleShape)
                    .background(BoneDeep),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = SOCIAL_PLATFORM_TILE[link.platform] ?: link.platform.take(2).uppercase(),
                    color = Slate,
                    style = Typography.labelMedium,
                    fontWeight = FontWeight.SemiBold,
                )
            }
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = SOCIAL_PLATFORM_LABEL[link.platform] ?: link.platform,
                    color = Ink,
                    style = Typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = link.url,
                    color = Slate,
                    style = Typography.bodySmall,
                    maxLines = 1,
                )
            }
            TextButton(onClick = onEdit) {
                Text(text = "EDIT", color = Ink, style = Typography.labelMedium)
            }
            TextButton(onClick = onDelete) {
                Text(text = "REMOVE", color = ErrorRed, style = Typography.labelMedium)
            }
        }
    }
}

// ─── Slab: Payouts ───────────────────────────────────────────────────────────

@Composable
private fun PayoutsSlab(
    payouts: List<PayoutAccountDto>,
    onAdd: () -> Unit,
    onEdit: (PayoutAccountDto) -> Unit,
    onAskDelete: (PayoutAccountDto) -> Unit,
    onSetPrimary: (PayoutAccountDto) -> Unit,
) {
    Column {
        Kicker(text = "Payouts · ${payouts.size} of ${PAYOUT_METHODS.size}", color = SlateSoft)
        Spacer(modifier = Modifier.height(12.dp))
        if (payouts.isEmpty()) {
            Text(
                text = "Add a payout method to receive your earnings.",
                color = SlateSoft,
                style = Typography.bodyMedium,
            )
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                payouts.forEach { account ->
                    PayoutRow(
                        account = account,
                        onEdit = { onEdit(account) },
                        onDelete = { onAskDelete(account) },
                        onSetPrimary = { onSetPrimary(account) },
                    )
                }
            }
        }
        Spacer(modifier = Modifier.height(12.dp))
        GhostCta(
            text = if (payouts.size >= PAYOUT_METHODS.size) "All methods added" else "Add payout method",
            onClick = onAdd,
            enabled = payouts.size < PAYOUT_METHODS.size,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun PayoutRow(
    account: PayoutAccountDto,
    onEdit: () -> Unit,
    onDelete: () -> Unit,
    onSetPrimary: () -> Unit,
) {
    QpCard(modifier = Modifier.fillMaxWidth(), padding = 12.dp) {
        Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(36.dp)
                        .clip(CircleShape)
                        .background(BoneDeep),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = (PAYOUT_METHOD_LABEL[account.method] ?: account.method).take(2).uppercase(),
                        color = Slate,
                        style = Typography.labelMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
                Spacer(modifier = Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = PAYOUT_METHOD_LABEL[account.method] ?: account.method,
                            color = Ink,
                            style = Typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                        if (account.isPrimary) {
                            Spacer(modifier = Modifier.width(8.dp))
                            StatusChip(text = "Primary", tone = StatusTone.Approved)
                        }
                    }
                    Text(
                        text = account.accountName,
                        color = Slate,
                        style = Typography.bodySmall,
                    )
                    Text(
                        text = formatPayoutNumber(account.method, account.accountNumber),
                        color = SlateSoft,
                        style = Typography.bodySmall,
                    )
                }
                // qr.dataUrl is named for the website's local pre-upload shape,
                // but the backend serves a presigned storage URL (PayoutAccount
                // Service.qrUrlFor), so Coil loads it directly — HostRewrite
                // Interceptor on the shared ImageLoader handles the dev
                // localhost host. Showing the actual code beats "on file": the
                // photographer can confirm they uploaded the right one.
                val qrUrl = account.qr?.dataUrl
                if (qrUrl != null) {
                    Spacer(modifier = Modifier.width(12.dp))
                    AsyncImage(
                        model = qrUrl,
                        contentDescription = "Payout QR code",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .size(44.dp)
                            .clip(QpCardShape)
                            .background(BoneDeep),
                    )
                }
            }
            Spacer(modifier = Modifier.height(6.dp))
            Row(modifier = Modifier.fillMaxWidth()) {
                if (!account.isPrimary) {
                    TextButton(onClick = onSetPrimary) {
                        Text(text = "MAKE PRIMARY", color = Ink, style = Typography.labelMedium)
                    }
                }
                Spacer(modifier = Modifier.weight(1f))
                TextButton(onClick = onEdit) {
                    Text(text = "EDIT", color = Ink, style = Typography.labelMedium)
                }
                TextButton(onClick = onDelete) {
                    Text(text = "REMOVE", color = ErrorRed, style = Typography.labelMedium)
                }
            }
        }
    }
}

// ─── Bottom actions ──────────────────────────────────────────────────────────

@Composable
private fun BottomActions(
    isSaving: Boolean,
    onSave: () -> Unit,
    onSignOut: () -> Unit,
) {
    Column {
        PrimaryCta(
            text = if (isSaving) "Saving…" else "Save changes",
            onClick = onSave,
            loading = isSaving,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(modifier = Modifier.height(10.dp))
        GhostCta(
            text = "Sign out",
            onClick = onSignOut,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

// ─── Reusable: text field + dropdown field ───────────────────────────────────

@Composable
private fun QpFormField(
    label: String,
    value: String,
    onChange: (String) -> Unit,
    placeholder: String = "",
    singleLine: Boolean = true,
    minLines: Int = 1,
    prefix: String? = null,
    keyboardType: KeyboardType = KeyboardType.Text,
    error: String? = null,
) {
    Column {
        Kicker(text = label, color = Slate)
        Spacer(modifier = Modifier.height(6.dp))
        OutlinedTextField(
            value = value,
            onValueChange = onChange,
            placeholder = { Text(placeholder, color = SlateSoft, style = Typography.bodyMedium) },
            prefix = if (prefix != null) {
                { Text(prefix, color = SlateSoft, style = Typography.bodyMedium) }
            } else null,
            singleLine = singleLine,
            minLines = minLines,
            keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
            isError = error != null,
            shape = FieldShape,
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Ink,
                unfocusedTextColor = Ink,
                focusedBorderColor = Ink,
                unfocusedBorderColor = Line,
                errorBorderColor = ErrorRed,
                cursorColor = Ink,
                focusedContainerColor = Bone,
                unfocusedContainerColor = Bone,
            ),
            textStyle = Typography.bodyMedium,
            modifier = Modifier.fillMaxWidth(),
        )
        if (error != null) {
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = error, color = ErrorRed, style = Typography.bodySmall)
        }
    }
}

@Composable
private fun DropdownField(
    label: String,
    valueLabel: String,
    placeholder: String,
    expanded: Boolean,
    onExpand: () -> Unit,
    onCollapse: () -> Unit,
    items: @Composable () -> Unit,
) {
    Column {
        Kicker(text = label, color = Slate)
        Spacer(modifier = Modifier.height(6.dp))
        Box {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(FieldShape)
                    .background(Bone)
                    .clickable { onExpand() }
                    .padding(horizontal = 14.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = valueLabel.ifBlank { placeholder },
                    color = if (valueLabel.isBlank()) SlateSoft else Ink,
                    style = Typography.bodyMedium,
                    modifier = Modifier.weight(1f),
                )
                Icon(
                    imageVector = Icons.Default.ArrowDropDown,
                    contentDescription = null,
                    tint = Slate,
                )
            }
            DropdownMenu(
                expanded = expanded,
                onDismissRequest = onCollapse,
            ) {
                items()
            }
        }
    }
}

// ─── Sheets: Social editor ───────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SocialEditorSheet(
    mode: SocialSheetMode,
    existingSocials: List<SocialLinkDto>,
    onDismiss: () -> Unit,
    onConfirm: (platform: String, url: String) -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    val availablePlatforms = remember(existingSocials, mode) {
        val taken = existingSocials.map { it.platform }.toSet()
        when (mode) {
            is SocialSheetMode.Add -> SOCIAL_PLATFORMS.filterNot { it in taken }
            is SocialSheetMode.Edit -> listOf(mode.link.platform)
        }
    }

    var platform by remember(mode) {
        mutableStateOf(
            when (mode) {
                is SocialSheetMode.Add -> availablePlatforms.firstOrNull() ?: "facebook"
                is SocialSheetMode.Edit -> mode.link.platform
            },
        )
    }
    var url by remember(mode) {
        mutableStateOf(if (mode is SocialSheetMode.Edit) mode.link.url else "")
    }
    var platformMenuExpanded by remember { mutableStateOf(false) }

    val validationError = remember(platform, url) { validateSocialUrl(platform, url) }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Bone,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
                .padding(bottom = 24.dp),
        ) {
            Kicker(
                text = if (mode is SocialSheetMode.Edit) "Edit social" else "Add social",
                color = Slate,
            )
            Spacer(modifier = Modifier.height(16.dp))

            DropdownField(
                label = "Platform",
                valueLabel = SOCIAL_PLATFORM_LABEL[platform] ?: platform,
                placeholder = "Pick a platform",
                expanded = platformMenuExpanded,
                onExpand = { if (mode is SocialSheetMode.Add) platformMenuExpanded = true },
                onCollapse = { platformMenuExpanded = false },
            ) {
                availablePlatforms.forEach { p ->
                    DropdownMenuItem(
                        text = { Text(SOCIAL_PLATFORM_LABEL[p] ?: p, color = Ink, style = Typography.bodyMedium) },
                        onClick = {
                            platform = p
                            platformMenuExpanded = false
                        },
                    )
                }
            }
            Spacer(modifier = Modifier.height(16.dp))

            QpFormField(
                label = "URL",
                value = url,
                onChange = { url = it },
                placeholder = "https://…",
                error = if (url.isNotBlank()) validationError else null,
            )

            Spacer(modifier = Modifier.height(20.dp))
            PrimaryCta(
                text = if (mode is SocialSheetMode.Edit) "Save changes" else "Add link",
                onClick = { onConfirm(platform, url.trim()) },
                enabled = url.trim().isNotBlank() && validationError == null,
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(modifier = Modifier.height(8.dp))
            GhostCta(
                text = "Cancel",
                onClick = onDismiss,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

// ─── Sheets: Payout editor ───────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PayoutEditorSheet(
    mode: PayoutSheetMode,
    onDismiss: () -> Unit,
    onConfirm: (method: String, accountName: String, accountNumber: String, qrBytes: ByteArray?) -> Unit,
) {
    val context = LocalContext.current
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    var method by remember(mode) {
        mutableStateOf(
            when (mode) {
                is PayoutSheetMode.Add -> "gcash"
                is PayoutSheetMode.Edit -> mode.account.method
            },
        )
    }
    var accountName by remember(mode) {
        mutableStateOf(if (mode is PayoutSheetMode.Edit) mode.account.accountName else "")
    }
    var accountNumberInput by remember(mode) {
        mutableStateOf(
            if (mode is PayoutSheetMode.Edit) formatPayoutNumber(mode.account.method, mode.account.accountNumber) else "",
        )
    }
    var qrBytes by remember(mode) { mutableStateOf<ByteArray?>(null) }
    var qrUri by remember(mode) { mutableStateOf<String?>(null) }
    var methodMenuExpanded by remember { mutableStateOf(false) }

    val qrPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
    ) { uri ->
        uri?.let {
            try {
                context.contentResolver.openInputStream(it)?.use { stream ->
                    qrBytes = stream.readBytes()
                    qrUri = it.toString()
                }
            } catch (e: Exception) {
                Toast.makeText(context, "Failed to load QR image", Toast.LENGTH_SHORT).show()
            }
        }
    }

    val numberDigits = accountNumberInput.filter { it.isDigit() }
    val numberError = remember(method, numberDigits) { validatePayoutNumber(method, numberDigits) }
    val numberMinSatisfied = when (method) {
        "gotyme" -> numberDigits.length in 10..16
        else -> numberDigits.length == 11 && numberDigits.startsWith("09")
    }
    val canSubmit = accountName.trim().isNotBlank() && numberMinSatisfied && numberError == null

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Bone,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
                .padding(bottom = 24.dp),
        ) {
            Kicker(
                text = if (mode is PayoutSheetMode.Edit) "Edit payout" else "Add payout",
                color = Slate,
            )
            Spacer(modifier = Modifier.height(16.dp))

            DropdownField(
                label = "Method",
                valueLabel = PAYOUT_METHOD_LABEL[method] ?: method,
                placeholder = "Pick a method",
                expanded = methodMenuExpanded,
                onExpand = { if (mode is PayoutSheetMode.Add) methodMenuExpanded = true },
                onCollapse = { methodMenuExpanded = false },
            ) {
                PAYOUT_METHODS.forEach { m ->
                    DropdownMenuItem(
                        text = { Text(PAYOUT_METHOD_LABEL[m] ?: m, color = Ink, style = Typography.bodyMedium) },
                        onClick = {
                            method = m
                            accountNumberInput = formatPayoutNumber(m, accountNumberInput.filter { it.isDigit() })
                            methodMenuExpanded = false
                        },
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            QpFormField(
                label = "Account name",
                value = accountName,
                onChange = { accountName = it },
                placeholder = "Full name on account",
            )

            Spacer(modifier = Modifier.height(16.dp))

            QpFormField(
                label = if (method == "gotyme") "Account number (16 digits)" else "Mobile number (09…)",
                value = accountNumberInput,
                onChange = { raw -> accountNumberInput = formatPayoutNumber(method, raw) },
                placeholder = if (method == "gotyme") "1234 5678 9012 3456" else "09XX XXX XXXX",
                keyboardType = KeyboardType.Number,
                error = if (numberDigits.isNotBlank()) numberError else null,
            )

            Spacer(modifier = Modifier.height(16.dp))

            Kicker(text = "QR code · optional", color = Slate)
            Spacer(modifier = Modifier.height(6.dp))
            QpCard(modifier = Modifier.fillMaxWidth(), padding = 12.dp) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(56.dp)
                            .clip(QpCardShape)
                            .background(BoneDeep),
                        contentAlignment = Alignment.Center,
                    ) {
                        val model = qrUri
                        if (model != null) {
                            AsyncImage(
                                model = model,
                                contentDescription = "Payout QR preview",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize(),
                            )
                        } else if (mode is PayoutSheetMode.Edit && mode.account.qr != null) {
                            // Stored QR — a presigned URL, so Coil renders it
                            // like any other remote image. See the card above.
                            AsyncImage(
                                model = mode.account.qr.dataUrl,
                                contentDescription = "Payout QR on file",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize(),
                            )
                        } else {
                            Text(text = "+", color = Slate, fontSize = 24.sp)
                        }
                    }
                    Spacer(modifier = Modifier.width(12.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = if (qrBytes != null) "Will replace on save" else "Pick a PNG of your QR",
                            color = Ink,
                            style = Typography.bodyMedium,
                            fontWeight = FontWeight.Medium,
                        )
                        Text(
                            text = "Shown to runners on the checkout receipt.",
                            color = SlateSoft,
                            style = Typography.bodySmall,
                        )
                    }
                    TextButton(onClick = {
                        qrPickerLauncher.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                        )
                    }) {
                        Text(text = "PICK", color = Ink, style = Typography.labelMedium)
                    }
                }
            }

            Spacer(modifier = Modifier.height(20.dp))
            PrimaryCta(
                text = if (mode is PayoutSheetMode.Edit) "Save changes" else "Add method",
                onClick = { onConfirm(method, accountName.trim(), numberDigits, qrBytes) },
                enabled = canSubmit,
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(modifier = Modifier.height(8.dp))
            GhostCta(
                text = "Cancel",
                onClick = onDismiss,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}
