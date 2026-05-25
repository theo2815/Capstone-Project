package com.quickpitik.mobile.ui.runner

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import android.content.ContentValues
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import androidx.compose.foundation.BorderStroke
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.ui.theme.*
import com.quickpitik.mobile.data.remote.*
import java.io.File
import java.io.FileOutputStream

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProfileScreen(
    viewModel: ProfileViewModel,
    cartViewModel: CartViewModel,
    savedEventsViewModel: SavedEventsViewModel,
    onNavigateBack: () -> Unit,
    onOpenEvent: (String) -> Unit,
    onBrowseEvents: () -> Unit
) {
    val selfies by viewModel.selfiesState.collectAsState()
    val isLoading by viewModel.selfiesLoading.collectAsState()
    val error by viewModel.selfiesError.collectAsState()
    val name by viewModel.profileName.collectAsState()
    val email by viewModel.profileEmail.collectAsState()

    // The race log is the union of saved (bookmarked) events and events the runner
    // has bought photos from — same as the website /profile race log. Orders come
    // from CartViewModel; saved events from the shared SavedEventsViewModel store.
    val ordersState by cartViewModel.ordersState.collectAsState()
    val savedEvents by savedEventsViewModel.savedEvents.collectAsState()

    // Trigger fetches if not loaded
    LaunchedEffect(Unit) {
        cartViewModel.fetchOrders()
        savedEventsViewModel.refresh()
        viewModel.fetchSelfies()
    }

    val orders = (ordersState as? OrdersState.Success)?.orders ?: emptyList()
    val raceLog = remember(orders, savedEvents) { buildRaceLog(savedEvents, orders) }
    val ordersLoading = ordersState is OrdersState.Loading

    val context = LocalContext.current
    var tempImageUri by remember { mutableStateOf<Uri?>(null) }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { viewModel.uploadSelfie(it) }
    }

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            tempImageUri?.let { uri ->
                viewModel.uploadSelfie(uri)
            }
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Bone
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
        ) {
            // Top Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(
                    onClick = onNavigateBack,
                    colors = IconButtonDefaults.iconButtonColors(containerColor = BoneDeep)
                ) {
                    Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = Ink)
                }
                Spacer(modifier = Modifier.width(16.dp))
                Text(
                    text = "RUNNER PROFILE",
                    style = Typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Ink
                )
            }

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 24.dp),
                verticalArrangement = Arrangement.spacedBy(24.dp)
            ) {
                // Identity Card
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(BoneDeep, RoundedCornerShape(16.dp))
                            .padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        // Avatar disc
                        Box(
                            modifier = Modifier
                                .size(72.dp)
                                .clip(CircleShape)
                                .background(Fresh),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = name.take(1).uppercase(),
                                color = Bone,
                                fontSize = 28.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = name,
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Ink,
                            textAlign = TextAlign.Center
                        )
                        Text(
                            text = email,
                            style = Typography.bodyMedium,
                            color = Slate,
                            textAlign = TextAlign.Center
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        SuggestionChip(
                            onClick = {},
                            label = { Text("RUNNER", color = Bone) },
                            colors = SuggestionChipDefaults.suggestionChipColors(containerColor = Ink),
                            border = null
                        )
                    }
                }

                // Selfie Library Section
                item {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text(
                                    text = "01. Selfie Library",
                                    style = Typography.titleMedium,
                                    fontWeight = FontWeight.Bold,
                                    color = Ink
                                )
                                Text(
                                    text = "Used for AI marathon face recognition",
                                    style = Typography.bodySmall,
                                    color = Slate
                                )
                            }
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Button(
                                    onClick = {
                                        try {
                                            val values = ContentValues().apply {
                                                put(MediaStore.Images.Media.TITLE, "captured_selfie_${System.currentTimeMillis()}")
                                                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                                            }
                                            val uri = context.contentResolver.insert(
                                                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                                                values
                                            )
                                            tempImageUri = uri
                                            if (uri != null) {
                                                cameraLauncher.launch(uri)
                                            }
                                        } catch (e: Exception) {
                                            // Handle exception
                                        }
                                    },
                                    colors = ButtonDefaults.buttonColors(containerColor = BoneDeep, contentColor = Ink),
                                    shape = RoundedCornerShape(8.dp),
                                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp),
                                    border = BorderStroke(1.dp, Line)
                                ) {
                                    Text("📷 CAMERA", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                }
                                Button(
                                    onClick = { galleryLauncher.launch("image/*") },
                                    colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                                    shape = RoundedCornerShape(8.dp),
                                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                                ) {
                                    Text("🖼️ GALLERY", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                }
                            }
                        }
                        
                        if (error != null) {
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = error ?: "",
                                color = ErrorRed,
                                style = Typography.bodySmall,
                                modifier = Modifier.fillMaxWidth()
                            )
                        }

                        if (isLoading) {
                            Spacer(modifier = Modifier.height(16.dp))
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(120.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                CircularProgressIndicator(color = Fresh)
                            }
                        } else if (selfies.isEmpty()) {
                            Spacer(modifier = Modifier.height(16.dp))
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .border(1.dp, SlateSoft, RoundedCornerShape(12.dp))
                                    .padding(24.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = "No selfies uploaded yet.\nUpload one to search marathon photos by face.",
                                    color = Slate,
                                    textAlign = TextAlign.Center,
                                    style = Typography.bodyMedium
                                )
                            }
                        } else {
                            Spacer(modifier = Modifier.height(16.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                selfies.forEach { selfie ->
                                    SelfieCard(
                                        selfie = selfie,
                                        onDelete = { viewModel.deleteSelfie(selfie.id) },
                                        onSetPrimary = { viewModel.setPrimarySelfie(selfie.id) },
                                        modifier = Modifier.weight(1f)
                                    )
                                }
                                // Fill empty weight slots to avoid visual bugs when 1 item exists
                                if (selfies.size < 3) {
                                    val blanks = 3 - selfies.size
                                    repeat(blanks) {
                                        Spacer(modifier = Modifier.weight(1f))
                                    }
                                }
                            }
                        }
                    }
                }

                // Race Log Section — saved ∪ purchased, deduped by event (web /profile)
                item {
                    Text(
                        text = "02. Race Log",
                        style = Typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = Ink
                    )
                    Text(
                        text = "Events you saved or bought photos from",
                        style = Typography.bodySmall,
                        color = Slate
                    )
                }

                if (ordersLoading && raceLog.isEmpty()) {
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(100.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator(color = Fresh)
                        }
                    }
                } else if (raceLog.isEmpty()) {
                    item {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .border(1.dp, SlateSoft, RoundedCornerShape(12.dp))
                                .padding(24.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(
                                text = "No races yet.",
                                color = Ink,
                                fontWeight = FontWeight.Bold,
                                style = Typography.bodyMedium
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Save a race or buy a photo and it'll show up here.",
                                color = Slate,
                                textAlign = TextAlign.Center,
                                style = Typography.bodySmall
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = "Browse races →",
                                color = Fresh,
                                fontWeight = FontWeight.Bold,
                                style = Typography.labelMedium,
                                modifier = Modifier.clickable { onBrowseEvents() }
                            )
                        }
                    }
                } else {
                    items(raceLog) { entry ->
                        RaceLogRow(
                            entry = entry,
                            onOpen = { entry.eventSlug?.let { onOpenEvent(it) } },
                            onUnsave = { savedEventsViewModel.unsave(entry.eventId, entry.eventName) }
                        )
                    }
                }

                item {
                    Spacer(modifier = Modifier.height(24.dp))
                }
            }
        }
    }
}

@Composable
fun SelfieCard(
    selfie: SelfieRefDto,
    onDelete: () -> Unit,
    onSetPrimary: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .aspectRatio(0.75f)
            .clip(RoundedCornerShape(12.dp))
            .background(BoneDeep)
            .clickable { if (!selfie.isPrimary) onSetPrimary() }
    ) {
        AsyncImage(
            model = selfie.dataUrl,
            contentDescription = "Runner Selfie",
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        // Overlay status badges
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(8.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Quality Score Badge
                Box(
                    modifier = Modifier
                        .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(4.dp))
                        .padding(horizontal = 4.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = "Q: ${(selfie.qualityScore * 100).toInt()}%",
                        color = Color.White,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                // Delete Button
                Box(
                    modifier = Modifier
                        .size(24.dp)
                        .clip(CircleShape)
                        .background(ErrorRed)
                        .clickable { onDelete() },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        Icons.Default.Delete,
                        contentDescription = "Delete",
                        tint = Bone,
                        modifier = Modifier.size(14.dp)
                    )
                }
            }

            // Primary Badge
            if (selfie.isPrimary) {
                Row(
                    modifier = Modifier
                        .background(Fresh, RoundedCornerShape(4.dp))
                        .padding(horizontal = 6.dp, vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    Icon(
                        Icons.Default.Star,
                        contentDescription = null,
                        tint = Bone,
                        modifier = Modifier.size(10.dp)
                    )
                    Text(
                        text = "PRIMARY",
                        color = Bone,
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}

// One de-duplicated race-log row: an event the runner saved, bought photos from,
// or both. Photo counts + spend come from the purchased side; the saved-only side
// contributes the bookmark + an Unsave affordance when the race is still upcoming.
private data class RaceLogEntry(
    val eventId: String,
    val eventName: String,
    val eventSlug: String?,
    val eventDate: String?,
    val photosBought: Int,
    val totalSpent: Double,
    val saved: Boolean,
    val purchased: Boolean
)

// saved ∪ purchased, keyed by eventId. Photos-bought and spend are summed across a
// runner's orders for the same event; saved flags the bookmark. Sorted newest-first
// by event date (nulls last). Mirrors the website's race-log derivation.
private fun buildRaceLog(
    saved: List<SavedEventSummaryDto>,
    orders: List<OrderListItemDto>
): List<RaceLogEntry> {
    val byEvent = LinkedHashMap<String, RaceLogEntry>()
    orders.forEach { order ->
        val existing = byEvent[order.eventId]
        byEvent[order.eventId] = RaceLogEntry(
            eventId = order.eventId,
            eventName = order.eventName ?: existing?.eventName ?: "Marathon Event",
            eventSlug = order.eventSlug ?: existing?.eventSlug,
            eventDate = order.eventDate ?: existing?.eventDate,
            photosBought = (existing?.photosBought ?: 0) + order.photoIds.size,
            totalSpent = (existing?.totalSpent ?: 0.0) + order.total,
            saved = existing?.saved ?: false,
            purchased = true
        )
    }
    saved.forEach { ev ->
        val existing = byEvent[ev.id]
        byEvent[ev.id] = RaceLogEntry(
            eventId = ev.id,
            eventName = existing?.eventName ?: ev.name,
            eventSlug = existing?.eventSlug ?: ev.slug,
            eventDate = existing?.eventDate ?: ev.date,
            photosBought = existing?.photosBought ?: 0,
            totalSpent = existing?.totalSpent ?: 0.0,
            saved = true,
            purchased = existing?.purchased ?: false
        )
    }
    return byEvent.values.sortedByDescending { it.eventDate ?: "" }
}

@Composable
private fun RaceLogRow(
    entry: RaceLogEntry,
    onOpen: () -> Unit,
    onUnsave: () -> Unit
) {
    val upcoming = entry.eventDate?.let { deriveEventState(it) == EventState.UPCOMING } ?: false
    val openable = !upcoming && entry.eventSlug != null
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(BoneDeep, RoundedCornerShape(12.dp))
            .then(if (openable) Modifier.clickable { onOpen() } else Modifier)
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = entry.eventName,
                style = Typography.titleSmall,
                fontWeight = FontWeight.Bold,
                color = Ink
            )
            Text(
                text = entry.eventDate?.let { eventDateLabel(it) } ?: "Date TBA",
                style = Typography.bodySmall,
                color = Slate
            )
            if (entry.purchased) {
                Text(
                    text = "${entry.photosBought} photo${if (entry.photosBought == 1) "" else "s"} kept",
                    style = Typography.bodySmall,
                    color = Fresh,
                    fontWeight = FontWeight.Bold
                )
            } else if (entry.saved) {
                Text(
                    text = if (upcoming) "Saved · photos coming on race day" else "Saved",
                    style = Typography.bodySmall,
                    color = SlateSoft
                )
            }
        }
        Column(horizontalAlignment = Alignment.End) {
            if (entry.purchased) {
                Text(
                    text = "₱%,.2f".format(entry.totalSpent),
                    style = Typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = Ink
                )
            }
            Spacer(modifier = Modifier.height(4.dp))
            if (upcoming && entry.saved && !entry.purchased) {
                Text(
                    text = "UNSAVE",
                    style = Typography.labelMedium,
                    color = ErrorRed,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.clickable { onUnsave() }
                )
            } else if (openable) {
                Text(
                    text = "Open →",
                    style = Typography.labelMedium,
                    color = Fresh,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}
