package com.quickpitik.mobile.ui.photographer

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.ui.theme.*
import java.io.ByteArrayOutputStream

data class PHProvince(val code: String, val name: String)
data class PHRegion(val code: String, val name: String, val provinces: List<PHProvince>)

val PH_REGIONS = listOf(
    PHRegion("ncr", "National Capital Region (NCR)", listOf(PHProvince("metro-manila", "Metro Manila"))),
    PHRegion("car", "Cordillera Administrative Region (CAR)", listOf(
        PHProvince("abra", "Abra"), PHProvince("apayao", "Apayao"), PHProvince("benguet", "Benguet"),
        PHProvince("ifugao", "Ifugao"), PHProvince("kalinga", "Kalinga"), PHProvince("mountain-province", "Mountain Province")
    )),
    PHRegion("region-1", "Region I (Ilocos Region)", listOf(
        PHProvince("ilocos-norte", "Ilocos Norte"), PHProvince("ilocos-sur", "Ilocos Sur"),
        PHProvince("la-union", "La Union"), PHProvince("pangasinan", "Pangasinan")
    )),
    PHRegion("region-2", "Region II (Cagayan Valley)", listOf(
        PHProvince("batanes", "Batanes"), PHProvince("cagayan", "Cagayan"),
        PHProvince("isabela", "Isabela"), PHProvince("nueva-vizcaya", "Nueva Vizcaya"), PHProvince("quirino", "Quirino")
    )),
    PHRegion("region-3", "Region III (Central Luzon)", listOf(
        PHProvince("aurora", "Aurora"), PHProvince("bataan", "Bataan"), PHProvince("bulacan", "Bulacan"),
        PHProvince("nueva-ecija", "Nueva Ecija"), PHProvince("pampanga", "Pampanga"),
        PHProvince("tarlac", "Tarlac"), PHProvince("zambales", "Zambales")
    )),
    PHRegion("region-4a", "Region IV-A (CALABARZON)", listOf(
        PHProvince("batangas", "Batangas"), PHProvince("cavite", "Cavite"),
        PHProvince("laguna", "Laguna"), PHProvince("quezon", "Quezon"), PHProvince("rizal", "Rizal")
    )),
    PHRegion("region-4b", "Region IV-B (MIMAROPA)", listOf(
        PHProvince("marinduque", "Marinduque"), PHProvince("occidental-mindoro", "Occidental Mindoro"),
        PHProvince("oriental-mindoro", "Oriental Mindoro"), PHProvince("palawan", "Palawan"), PHProvince("romblon", "Romblon")
    )),
    PHRegion("region-5", "Region V (Bicol Region)", listOf(
        PHProvince("albay", "Albay"), PHProvince("camarines-norte", "Camarines Norte"),
        PHProvince("camarines-sur", "Camarines Sur"), PHProvince("catanduanes", "Catanduanes"),
        PHProvince("masbate", "Masbate"), PHProvince("sorsogon", "Sorsogon")
    )),
    PHRegion("region-6", "Region VI (Western Visayas)", listOf(
        PHProvince("aklan", "Aklan"), PHProvince("antique", "Antique"), PHProvince("capiz", "Capiz"),
        PHProvince("guimaras", "Guimaras"), PHProvince("iloilo", "Iloilo"), PHProvince("negros-occidental", "Negros Occidental")
    )),
    PHRegion("region-7", "Region VII (Central Visayas)", listOf(
        PHProvince("bohol", "Bohol"), PHProvince("cebu", "Cebu"),
        PHProvince("negros-oriental", "Negros Oriental"), PHProvince("siquijor", "Siquijor")
    )),
    PHRegion("region-8", "Region VIII (Eastern Visayas)", listOf(
        PHProvince("biliran", "Biliran"), PHProvince("leyte", "Leyte"), PHProvince("northern-samar", "Northern Samar"),
        PHProvince("samar", "Samar"), PHProvince("southern-leyte", "Southern Leyte"), PHProvince("eastern-samar", "Eastern Samar")
    )),
    PHRegion("region-9", "Region IX (Zamboanga Peninsula)", listOf(
        PHProvince("zamboanga-del-norte", "Zamboanga del Norte"),
        PHProvince("zamboanga-del-sur", "Zamboanga del Sur"),
        PHProvince("zamboanga-sibugay", "Zamboanga Sibugay")
    )),
    PHRegion("region-10", "Region X (Northern Mindanao)", listOf(
        PHProvince("bukidnon", "Bukidnon"), PHProvince("camiguin", "Camiguin"),
        PHProvince("lanao-del-norte", "Lanao del Norte"), PHProvince("misamis-occidental", "Misamis Occidental"),
        PHProvince("misamis-oriental", "Misamis Oriental")
    )),
    PHRegion("region-11", "Region XI (Davao Region)", listOf(
        PHProvince("davao-de-oro", "Davao de Oro"), PHProvince("davao-del-norte", "Davao del Norte"),
        PHProvince("davao-del-sur", "Davao del Sur"), PHProvince("davao-occidental", "Davao Occidental"),
        PHProvince("davao-oriental", "Davao Oriental")
    )),
    PHRegion("region-12", "Region XII (SOCCSKSARGEN)", listOf(
        PHProvince("cotabato", "Cotabato"), PHProvince("sarangani", "Sarangani"),
        PHProvince("south-cotabato", "South Cotabato"), PHProvince("sultan-kudarat", "Sultan Kudarat")
    )),
    PHRegion("region-13", "Region XIII (Caraga)", listOf(
        PHProvince("agusan-del-norte", "Agusan del Norte"), PHProvince("agusan-del-sur", "Agusan del Sur"),
        PHProvince("dinagat-islands", "Dinagat Islands"), PHProvince("surigao-del-norte", "Surigao del Norte"),
        PHProvince("surigao-del-sur", "Surigao del Sur")
    )),
    PHRegion("barmm", "Bangsamoro Autonomous Region in Muslim Mindanao (BARMM)", listOf(
        PHProvince("basilan", "Basilan"), PHProvince("lanao-del-sur", "Lanao del Sur"),
        PHProvince("maguindanao-del-norte", "Maguindanao del Norte"), PHProvince("maguindanao-del-sur", "Maguindanao del Sur"),
        PHProvince("sulu", "Sulu"), PHProvince("tawi-tawi", "Tawi-Tawi")
    ))
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotographerSettingsScreen(
    viewModel: PhotographerDashboardViewModel,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val verificationState by viewModel.verificationState.collectAsState()
    val brandSettings by viewModel.brandSettings.collectAsState()
    val payoutAccounts by viewModel.payoutAccounts.collectAsState()
    val socials by viewModel.socials.collectAsState()
    val isSaving by viewModel.isSavingSettings.collectAsState()
    val actionMessage by viewModel.settingsActionState.collectAsState()

    // Form inputs state
    var brandName by remember { mutableStateOf("") }
    var bio by remember { mutableStateOf("") }
    var gcashName by remember { mutableStateOf("") }
    var gcashNumber by remember { mutableStateOf("") }
    var handle by remember { mutableStateOf("") }
    var regionCode by remember { mutableStateOf("") }
    var provinceCode by remember { mutableStateOf("") }
    var socialUrl by remember { mutableStateOf("") }

    // Dropdown visibility state
    var regionDropdownExpanded by remember { mutableStateOf(false) }
    var provinceDropdownExpanded by remember { mutableStateOf(false) }

    val currentSelectedRegion = PH_REGIONS.firstOrNull { it.code == regionCode }
    val currentSelectedProvince = currentSelectedRegion?.provinces?.firstOrNull { it.code == provinceCode }

    // Upload staged states
    var avatarUri by remember { mutableStateOf<String?>(null) }
    var coverUri by remember { mutableStateOf<String?>(null) }
    var watermarkUri by remember { mutableStateOf<String?>(null) }

    var avatarBytes by remember { mutableStateOf<ByteArray?>(null) }
    var coverBytes by remember { mutableStateOf<ByteArray?>(null) }
    var watermarkBytes by remember { mutableStateOf<ByteArray?>(null) }

    // Image Picker launchers
    val avatarPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
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
        contract = ActivityResultContracts.PickVisualMedia()
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
        contract = ActivityResultContracts.PickVisualMedia()
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

    // Display action toasts on flow states
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

    // Pre-populate fields once settings load successfully
    LaunchedEffect(brandSettings) {
        brandSettings?.let {
            if (brandName.isEmpty()) brandName = it.brandName.orEmpty()
            if (bio.isEmpty()) bio = it.bio.orEmpty()
            if (handle.isEmpty()) handle = it.handle.orEmpty()
            if (regionCode.isEmpty()) regionCode = it.regionCode.orEmpty()
            if (provinceCode.isEmpty()) provinceCode = it.provinceCode.orEmpty()
        }
    }

    LaunchedEffect(payoutAccounts) {
        if (payoutAccounts.isNotEmpty()) {
            val primaryPayout = payoutAccounts.firstOrNull { it.isPrimary } ?: payoutAccounts.first()
            if (gcashName.isEmpty()) gcashName = primaryPayout.accountName
            if (gcashNumber.isEmpty()) gcashNumber = primaryPayout.accountNumber
        }
    }

    LaunchedEffect(socials) {
        if (socials.isNotEmpty() && socialUrl.isEmpty()) {
            socialUrl = socials.firstOrNull { it.platform.lowercase() == "facebook" }?.url
                ?: socials.first().url
        }
    }

    var selectedTabIndex by remember { mutableStateOf(0) }
    val tabTitles = listOf("Brand Profile", "Payout", "Verification")

    val scrollState = rememberScrollState()

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Bone)
            .verticalScroll(scrollState)
    ) {
        // --- 1. HERO HEADER AREA ---
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(180.dp)
        ) {
            // Cover Image
            val resolvedCoverUrl = brandSettings?.coverUrl?.replace("localhost", "10.0.2.2")
            if (coverUri != null) {
                AsyncImage(
                    model = coverUri,
                    contentDescription = "Cover Image Preview",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            } else if (!resolvedCoverUrl.isNullOrEmpty()) {
                AsyncImage(
                    model = resolvedCoverUrl,
                    contentDescription = "Cover Image",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            Brush.linearGradient(
                                colors = listOf(SlateSoft, BoneDeep)
                            )
                        )
                )
            }

            // Dark overlay for readability
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.25f))
            )

            // Overlapping Profile Avatar
            val resolvedAvatarUrl = brandSettings?.avatarUrl?.replace("localhost", "10.0.2.2")
            Box(
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(start = 20.dp, bottom = 12.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(72.dp)
                            .clip(CircleShape)
                            .background(BoneDeep)
                            .border(3.dp, Color.White, CircleShape)
                    ) {
                        if (avatarUri != null) {
                            AsyncImage(
                                model = avatarUri,
                                contentDescription = "Avatar Preview",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                            )
                        } else if (!resolvedAvatarUrl.isNullOrEmpty()) {
                            AsyncImage(
                                model = resolvedAvatarUrl,
                                contentDescription = "Avatar",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                            )
                        } else {
                            Icon(
                                imageVector = Icons.Default.Person,
                                contentDescription = "No Avatar",
                                tint = SlateSoft,
                                modifier = Modifier
                                    .size(36.dp)
                                    .align(Alignment.Center)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.width(12.dp))

                    Column {
                        Text(
                            text = brandName.ifBlank { "Your Studio Name" },
                            color = Color.White,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "@${handle.ifBlank { "studio" }}",
                            color = Color.White.copy(alpha = 0.8f),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
            }
        }

        // --- 2. VERIFICATION STATUS HEADER ---
        PaddingContainer {
            VerificationStatusHeader(
                verificationState = verificationState,
                onSubmit = { viewModel.submitVerification() },
                onWithdraw = { viewModel.withdrawVerification() }
            )
        }

        // --- 3. PREMIUM TAB ROW ---
        TabRow(
            selectedTabIndex = selectedTabIndex,
            containerColor = BoneDeep,
            contentColor = Fresh,
            indicator = { tabPositions ->
                TabRowDefaults.Indicator(
                    modifier = Modifier.tabIndicatorOffset(tabPositions[selectedTabIndex]),
                    color = Fresh,
                    height = 3.dp
                )
            },
            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)
        ) {
            tabTitles.forEachIndexed { index, title ->
                Tab(
                    selected = selectedTabIndex == index,
                    onClick = { selectedTabIndex = index },
                    text = {
                        Text(
                            text = title,
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp,
                            color = if (selectedTabIndex == index) Fresh else SlateSoft
                        )
                    },
                    icon = {
                        Icon(
                            imageVector = when (index) {
                                0 -> Icons.Default.AccountBox
                                1 -> Icons.Default.ShoppingCart
                                else -> Icons.Default.CheckCircle
                            },
                            contentDescription = title,
                            tint = if (selectedTabIndex == index) Fresh else SlateSoft,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                )
            }
        }

        // --- 4. TAB CONTENTS ---
        PaddingContainer {
            when (selectedTabIndex) {
                0 -> {
                    // TAB 0: Brand Profile
                    SlabContainer(
                        title = "Brand & Presentation",
                        kicker = "Profile Details"
                    ) {
                        OutlinedTextField(
                            value = brandName,
                            onValueChange = { brandName = it },
                            label = { Text("Brand Name / Studio Name", color = SlateSoft) },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Fresh,
                                unfocusedBorderColor = SlateSoft.copy(alpha = 0.5f),
                                focusedTextColor = Ink,
                                unfocusedTextColor = Ink,
                                focusedContainerColor = BoneDeep,
                                unfocusedContainerColor = BoneDeep
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        OutlinedTextField(
                            value = handle,
                            onValueChange = { handle = it },
                            label = { Text("Public Handle (e.g. studio-handle)", color = SlateSoft) },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Fresh,
                                unfocusedBorderColor = SlateSoft.copy(alpha = 0.5f),
                                focusedTextColor = Ink,
                                unfocusedTextColor = Ink,
                                focusedContainerColor = BoneDeep,
                                unfocusedContainerColor = BoneDeep
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(16.dp))

                        // Region Dropdown stacked vertically for generous mobile legibility
                        Box(modifier = Modifier.fillMaxWidth()) {
                            OutlinedTextField(
                                value = currentSelectedRegion?.name ?: "Pick a region",
                                onValueChange = {},
                                readOnly = true,
                                label = { Text("Region", color = SlateSoft) },
                                trailingIcon = {
                                    Icon(
                                        imageVector = Icons.Default.ArrowDropDown,
                                        contentDescription = "Drop Down",
                                        tint = SlateSoft,
                                        modifier = Modifier.clickable { regionDropdownExpanded = true }
                                    )
                                },
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = Fresh,
                                    unfocusedBorderColor = SlateSoft.copy(alpha = 0.5f),
                                    focusedTextColor = Ink,
                                    unfocusedTextColor = Ink,
                                    focusedContainerColor = BoneDeep,
                                    unfocusedContainerColor = BoneDeep
                                ),
                                modifier = Modifier.fillMaxWidth().clickable { regionDropdownExpanded = true }
                            )
                            DropdownMenu(
                                expanded = regionDropdownExpanded,
                                onDismissRequest = { regionDropdownExpanded = false },
                                modifier = Modifier.fillMaxWidth(0.85f).height(300.dp).background(BoneDeep)
                            ) {
                                PH_REGIONS.forEach { reg ->
                                    DropdownMenuItem(
                                        text = { Text(reg.name, color = Ink, fontSize = 13.sp) },
                                        onClick = {
                                            regionCode = reg.code
                                            provinceCode = ""
                                            regionDropdownExpanded = false
                                        }
                                    )
                                }
                            }
                        }
                        
                        Spacer(modifier = Modifier.height(16.dp))

                        // Province Dropdown stacked vertically
                        Box(modifier = Modifier.fillMaxWidth()) {
                            OutlinedTextField(
                                value = currentSelectedProvince?.name ?: if (provinceCode.isNotEmpty()) provinceCode else "Pick a province",
                                onValueChange = {},
                                readOnly = true,
                                label = { Text("Province", color = SlateSoft) },
                                trailingIcon = {
                                    Icon(
                                        imageVector = Icons.Default.ArrowDropDown,
                                        contentDescription = "Drop Down",
                                        tint = SlateSoft,
                                        modifier = Modifier.clickable { provinceDropdownExpanded = true }
                                    )
                                },
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = Fresh,
                                    unfocusedBorderColor = SlateSoft.copy(alpha = 0.5f),
                                    focusedTextColor = Ink,
                                    unfocusedTextColor = Ink,
                                    focusedContainerColor = BoneDeep,
                                    unfocusedContainerColor = BoneDeep
                                ),
                                modifier = Modifier.fillMaxWidth().clickable { provinceDropdownExpanded = true }
                            )
                            DropdownMenu(
                                expanded = provinceDropdownExpanded,
                                onDismissRequest = { provinceDropdownExpanded = false },
                                modifier = Modifier.fillMaxWidth(0.85f).height(300.dp).background(BoneDeep)
                            ) {
                                val provinces = currentSelectedRegion?.provinces ?: emptyList()
                                if (provinces.isEmpty()) {
                                    DropdownMenuItem(
                                        text = { Text("Select region first", color = SlateSoft) },
                                        onClick = { provinceDropdownExpanded = false }
                                    )
                                } else {
                                    provinces.forEach { prov ->
                                        DropdownMenuItem(
                                            text = { Text(prov.name, color = Ink, fontSize = 13.sp) },
                                            onClick = {
                                                provinceCode = prov.code
                                                provinceDropdownExpanded = false
                                            }
                                        )
                                    }
                                }
                            }
                        }
                        
                        Spacer(modifier = Modifier.height(16.dp))

                        OutlinedTextField(
                            value = socialUrl,
                            onValueChange = { socialUrl = it },
                            label = { Text("Facebook Social Profile URL", color = SlateSoft) },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Fresh,
                                unfocusedBorderColor = SlateSoft.copy(alpha = 0.5f),
                                focusedTextColor = Ink,
                                unfocusedTextColor = Ink,
                                focusedContainerColor = BoneDeep,
                                unfocusedContainerColor = BoneDeep
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(16.dp))

                        OutlinedTextField(
                            value = bio,
                            onValueChange = { bio = it },
                            label = { Text("Brand Biography", color = SlateSoft) },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Fresh,
                                unfocusedBorderColor = SlateSoft.copy(alpha = 0.5f),
                                focusedTextColor = Ink,
                                unfocusedTextColor = Ink,
                                focusedContainerColor = BoneDeep,
                                unfocusedContainerColor = BoneDeep
                            ),
                            modifier = Modifier.fillMaxWidth().height(120.dp)
                        )
                        
                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = "Describe your unique style, experience, and gear to prospective runners buying your work.",
                            color = SlateSoft,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
                1 -> {
                    // TAB 1: Payout Details
                    SlabContainer(
                        title = "Payout Configuration",
                        kicker = "Fast GCash"
                    ) {
                        OutlinedTextField(
                            value = gcashName,
                            onValueChange = { gcashName = it },
                            label = { Text("Account Holder Full Name", color = SlateSoft) },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Fresh,
                                unfocusedBorderColor = SlateSoft.copy(alpha = 0.5f),
                                focusedTextColor = Ink,
                                unfocusedTextColor = Ink,
                                focusedContainerColor = BoneDeep,
                                unfocusedContainerColor = BoneDeep
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(18.dp))
                        OutlinedTextField(
                            value = gcashNumber,
                            onValueChange = { gcashNumber = it },
                            label = { Text("GCash Mobile Number (11-Digits)", color = SlateSoft) },
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Fresh,
                                unfocusedBorderColor = SlateSoft.copy(alpha = 0.5f),
                                focusedTextColor = Ink,
                                unfocusedTextColor = Ink,
                                focusedContainerColor = BoneDeep,
                                unfocusedContainerColor = BoneDeep
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )
                        
                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = "Revenue payouts are automated directly into your primary GCash account at the end of each publishing cycle.",
                            color = SlateSoft,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
                2 -> {
                    // TAB 2: Verification Assets
                    SlabContainer(
                        title = "Identity & Protection Assets",
                        kicker = "Interactive Media"
                    ) {
                        Text(
                            text = "Watermarks protect your copyrighted work. Tap any slot to select your real image files dynamically.",
                            color = SlateSoft,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium,
                            modifier = Modifier.padding(bottom = 16.dp)
                        )

                        // Avatar
                        VisualUploadCard(
                            label = "Profile Picture (Avatar)",
                            kicker = "Tapped to update your circular avatar badge",
                            selectedUri = avatarUri,
                            existingUrl = brandSettings?.avatarUrl,
                            placeholderIcon = Icons.Default.Person,
                            onClick = {
                                avatarPickerLauncher.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                )
                            },
                            aspectRatio = 3.5f,
                            isCircle = true
                        )

                        Spacer(modifier = Modifier.height(18.dp))

                        // Cover
                        VisualUploadCard(
                            label = "Cover Banner Image",
                            kicker = "Showcases your landscape banner atop your profile",
                            selectedUri = coverUri,
                            existingUrl = brandSettings?.coverUrl,
                            placeholderIcon = Icons.Default.Add,
                            onClick = {
                                coverPickerLauncher.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                )
                            },
                            aspectRatio = 2.5f
                        )

                        Spacer(modifier = Modifier.height(18.dp))

                        // Watermark
                        VisualUploadCard(
                            label = "DSLR Transparent Watermark",
                            kicker = "Upload transparent PNG files to watermark DSLR syncs",
                            selectedUri = watermarkUri,
                            existingUrl = brandSettings?.watermarkUrl,
                            placeholderIcon = Icons.Default.Star,
                            onClick = {
                                watermarkPickerLauncher.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                )
                            },
                            aspectRatio = 3f
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            Button(
                onClick = {
                    val validationError = when {
                        // 1. Brand Profile Validation
                        brandName.isBlank() -> "Brand Name/Studio Name is required"
                        handle.isBlank() -> "Public Handle is required"
                        regionCode.isBlank() -> "Please select a Region"
                        provinceCode.isBlank() -> "Please select a Province"
                        socialUrl.isBlank() -> "Facebook Social Profile URL is required"
                        bio.isBlank() -> "Brand Biography is required"
                        
                        // 2. Payout Validation
                        gcashName.isBlank() -> "GCash Account Holder Name is required"
                        gcashNumber.isBlank() -> "GCash Mobile Number is required"
                        gcashNumber.length != 11 || !gcashNumber.all { it.isDigit() } -> "GCash Mobile Number must be exactly 11 digits (e.g., 09123456789)"
                        
                        // 3. Verification Images Validation
                        (avatarUri == null && brandSettings?.avatarUrl.isNullOrEmpty()) -> "Please upload a Profile Picture (Avatar)"
                        (coverUri == null && brandSettings?.coverUrl.isNullOrEmpty()) -> "Please upload a Cover Banner Image"
                        (watermarkUri == null && brandSettings?.watermarkUrl.isNullOrEmpty()) -> "Please upload a DSLR Transparent Watermark"
                        
                        else -> null
                    }

                    if (validationError != null) {
                        Toast.makeText(context, validationError, Toast.LENGTH_LONG).show()
                    } else {
                        viewModel.saveSettings(
                            brandName = brandName,
                            bio = bio,
                            gcashName = gcashName,
                            gcashNumber = gcashNumber,
                            handle = handle,
                            regionCode = regionCode,
                            provinceCode = provinceCode,
                            socialUrl = socialUrl,
                            avatarBytes = avatarBytes,
                            coverBytes = coverBytes,
                            watermarkBytes = watermarkBytes
                        )
                    }
                },
                colors = ButtonDefaults.buttonColors(containerColor = Fresh),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                enabled = !isSaving
            ) {
                if (isSaving) {
                    CircularProgressIndicator(color = Color.White, modifier = Modifier.size(24.dp))
                } else {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.Check,
                            contentDescription = "Save",
                            tint = Color.White,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("SAVE SETTINGS CHANGES", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(40.dp))
    }
}

@Composable
fun PaddingContainer(
    content: @Composable ColumnScope.() -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp)
    ) {
        content()
    }
}

@Composable
fun VerificationStatusHeader(
    verificationState: VerificationUiState,
    onSubmit: () -> Unit,
    onWithdraw: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier
            .fillMaxWidth()
            .border(BorderStroke(1.dp, Line), RoundedCornerShape(16.dp))
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Text(
                text = "ONBOARDING VERIFICATION",
                color = SlateSoft,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.sp
            )
            Spacer(modifier = Modifier.height(8.dp))

            when (verificationState) {
                is VerificationUiState.Loading -> {
                    Box(
                        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator(color = Fresh, modifier = Modifier.size(24.dp))
                    }
                }
                is VerificationUiState.Error -> {
                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = "Error",
                            tint = ErrorRed,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Error: ${verificationState.message}",
                            color = ErrorRed,
                            fontSize = 13.sp
                        )
                    }
                }
                is VerificationUiState.Success -> {
                    val dto = verificationState.verification
                    val status = dto.status.lowercase()

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    imageVector = when (status) {
                                        "approved" -> Icons.Default.CheckCircle
                                        "pending" -> Icons.Default.Info
                                        "rejected" -> Icons.Default.Warning
                                        else -> Icons.Default.Warning
                                    },
                                    contentDescription = status,
                                    tint = when (status) {
                                        "approved" -> Fresh
                                        "pending" -> WarningOrange
                                        "rejected" -> ErrorRed
                                        else -> ErrorRed
                                    },
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = status.uppercase(),
                                    color = when (status) {
                                        "approved" -> Fresh
                                        "pending" -> WarningOrange
                                        "rejected" -> ErrorRed
                                        else -> ErrorRed
                                    },
                                    fontSize = 15.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }

                            if (status == "incomplete" && !dto.missing.isNullOrEmpty()) {
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "Missing: ${dto.missing.joinToString(", ")}",
                                    color = SlateSoft,
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }

                        Spacer(modifier = Modifier.width(12.dp))

                        // Submit or Withdraw Buttons
                        when (status) {
                            "incomplete" -> {
                                Button(
                                    onClick = onSubmit,
                                    colors = ButtonDefaults.buttonColors(containerColor = Fresh),
                                    shape = RoundedCornerShape(8.dp),
                                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 8.dp)
                                ) {
                                    Text("SUBMIT REVIEW", fontSize = 11.sp, color = Color.White, fontWeight = FontWeight.Bold)
                                }
                            }
                            "pending" -> {
                                Button(
                                    onClick = onWithdraw,
                                    colors = ButtonDefaults.buttonColors(containerColor = Line),
                                    shape = RoundedCornerShape(8.dp),
                                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 8.dp)
                                ) {
                                    Text("RESCIND REVIEW", fontSize = 11.sp, color = Ink, fontWeight = FontWeight.Bold)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun SlabContainer(
    title: String,
    kicker: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier
            .fillMaxWidth()
            .border(BorderStroke(1.dp, Line), RoundedCornerShape(16.dp))
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(text = title, color = Ink, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                Box(
                    modifier = Modifier
                        .background(Fresh.copy(alpha = 0.12f), RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 3.dp)
                ) {
                    Text(text = kicker, color = Fresh, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                }
            }
            content()
        }
    }
}

@Composable
fun VisualUploadCard(
    label: String,
    kicker: String,
    selectedUri: String?,
    existingUrl: String?,
    placeholderIcon: ImageVector,
    onClick: () -> Unit,
    aspectRatio: Float = 2.5f,
    isCircle: Boolean = false
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = Bone),
        border = BorderStroke(1.dp, Line),
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() }
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f).padding(end = 12.dp)) {
                    Text(text = label, color = Ink, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(text = kicker, color = SlateSoft, fontSize = 11.sp)
                }
                
                Surface(
                    shape = RoundedCornerShape(6.dp),
                    color = if (selectedUri != null) Fresh.copy(alpha = 0.12f) else Line.copy(alpha = 0.6f),
                    modifier = Modifier.wrapContentSize()
                ) {
                    Text(
                        text = if (selectedUri != null) "NEW IMAGE" else if (!existingUrl.isNullOrEmpty()) "ACTIVE ON BACKEND" else "EMPTY",
                        color = if (selectedUri != null) Fresh else SlateSoft,
                        fontSize = 9.sp,
                        fontWeight = FontWeight.ExtraBold,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        maxLines = 1
                    )
                }
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(aspectRatio)
                    .clip(if (isCircle) CircleShape else RoundedCornerShape(12.dp))
                    .background(BoneDeep)
                    .border(1.dp, Line, if (isCircle) CircleShape else RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                if (selectedUri != null) {
                    AsyncImage(
                        model = selectedUri,
                        contentDescription = label,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                } else if (!existingUrl.isNullOrEmpty()) {
                    val resolvedUrl = existingUrl.replace("localhost", "10.0.2.2")
                    AsyncImage(
                        model = resolvedUrl,
                        contentDescription = label,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(
                            imageVector = placeholderIcon,
                            contentDescription = "Placeholder",
                            tint = SlateSoft,
                            modifier = Modifier.size(32.dp)
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text("Tap to Select Image File", color = SlateSoft, fontSize = 11.sp, fontWeight = FontWeight.Medium)
                    }
                }
            }
        }
    }
}
