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

    val hapticFire = rememberQpHaptic()
    val snackbarHostState = remember { SnackbarHostState() }
    var snackbarMessage by remember { mutableStateOf<String?>(null) }
    LaunchedEffect(snackbarMessage) {
        snackbarMessage?.let { msg ->
            snackbarHostState.showSnackbar(msg)
            snackbarMessage = null
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Bone)) {
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
                Kicker("Runner profile")
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
                            .background(BoneDeep, QpCardShape)
                            .border(1.dp, Line, QpCardShape)
                            .padding(horizontal = 24.dp, vertical = 28.dp),
                    ) {
                        Kicker("Runner")
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = name,
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Ink,
                        )
                        Spacer(modifier = Modifier.height(2.dp))
                        Text(
                            text = email,
                            style = Typography.bodyMedium,
                            color = Slate,
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
                                Kicker("01 · Selfie library")
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "Used for AI face recognition.",
                                    style = Typography.bodySmall,
                                    color = Slate
                                )
                            }
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                OutlinedButton(
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
                                    shape = PillShape,
                                    border = BorderStroke(1.dp, Ink),
                                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink),
                                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp),
                                ) {
                                    Text("Camera", style = Typography.labelMedium, fontWeight = FontWeight.SemiBold)
                                }
                                Button(
                                    onClick = { galleryLauncher.launch("image/*") },
                                    colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Color.White),
                                    shape = PillShape,
                                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp)
                                ) {
                                    Text("Gallery", style = Typography.labelMedium, fontWeight = FontWeight.SemiBold)
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
                            SelfieRowSkeleton()
                        } else if (selfies.isEmpty()) {
                            Spacer(modifier = Modifier.height(16.dp))
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .border(1.dp, Line, FieldShape)
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
                                    key(selfie.id) {
                                        SelfieCard(
                                            selfie = selfie,
                                            onDelete = { viewModel.deleteSelfie(selfie.id) },
                                            onSetPrimary = {
                                                hapticFire(QpHaptic.CONFIRM)
                                                viewModel.setPrimarySelfie(selfie.id)
                                                snackbarMessage = "Primary selfie updated."
                                            },
                                            modifier = Modifier.weight(1f)
                                        )
                                    }
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
                    Column {
                        Kicker("02 · Race log")
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Events you saved or bought photos from.",
                            style = Typography.bodySmall,
                            color = Slate
                        )
                    }
                }

                if (ordersLoading && raceLog.isEmpty()) {
                    item {
                        RaceLogSkeleton()
                    }
                } else if (raceLog.isEmpty()) {
                    item {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .border(1.dp, Line, FieldShape)
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
                            ArrowLabel(
                                text = "Browse races →",
                                color = Fresh,
                                fontWeight = FontWeight.Bold,
                                style = Typography.labelMedium,
                                modifier = Modifier.clickable { onBrowseEvents() }
                            )
                        }
                    }
                } else {
                    items(raceLog, key = { it.eventId }) { entry ->
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
        SnackbarHost(
            hostState = snackbarHostState,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp),
        )
    }
}

@Composable
private fun SelfieRowSkeleton() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        repeat(3) {
            LoadingSkeleton(
                shape = QpCardShape,
                modifier = Modifier
                    .weight(1f)
                    .aspectRatio(0.75f),
            )
        }
    }
}

@Composable
private fun RaceLogSkeleton() {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        repeat(3) {
            LoadingSkeleton(
                shape = QpCardShape,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(88.dp),
            )
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
            .clip(QpCardShape)
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
                // Quality Score Badge — slate scrim, not pure black
                Box(
                    modifier = Modifier
                        .background(Ink.copy(alpha = 0.7f), BadgeShape)
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = "Q ${(selfie.qualityScore * 100).toInt()}%",
                        style = Typography.labelSmall,
                        color = Color.White,
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
                        tint = Color.White,
                        modifier = Modifier.size(14.dp)
                    )
                }
            }

            // Primary affordance — Fresh badge when primary, Slate "Set primary" hint otherwise
            if (selfie.isPrimary) {
                Row(
                    modifier = Modifier
                        .background(Fresh, BadgeShape)
                        .padding(horizontal = 6.dp, vertical = 3.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(3.dp)
                ) {
                    Icon(
                        Icons.Default.Star,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(12.dp)
                    )
                    Text(
                        text = "PRIMARY",
                        style = Typography.labelSmall,
                        color = Color.White,
                    )
                }
            } else {
                Box(
                    modifier = Modifier
                        .background(Slate.copy(alpha = 0.85f), BadgeShape)
                        .padding(horizontal = 6.dp, vertical = 3.dp),
                ) {
                    Text(
                        text = "TAP TO SET",
                        style = Typography.labelSmall,
                        color = Color.White,
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
            .background(BoneDeep, QpCardShape)
            .border(1.dp, Line, QpCardShape)
            .then(if (openable) Modifier.clickable { onOpen() } else Modifier)
            .padding(horizontal = 16.dp, vertical = 18.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f)) {
            // Date first — kicker style, mono tnum (website parity)
            Kicker(entry.eventDate?.let { eventDateLabel(it) } ?: "Date TBA")
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = entry.eventName,
                style = Typography.titleMedium,
                color = Ink,
                maxLines = 2,
            )
            Spacer(modifier = Modifier.height(6.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = when {
                        entry.purchased -> "Photos kept"
                        upcoming && entry.saved -> "Saved · photos on race day"
                        entry.saved -> "Saved"
                        else -> "Archived"
                    },
                    style = Typography.bodySmall,
                    color = Slate,
                )
                if (entry.purchased && entry.photosBought > 0) {
                    Text(
                        text = "  ·  ",
                        style = Typography.bodySmall,
                        color = SlateSoft,
                    )
                    Text(
                        text = "${entry.photosBought}",
                        style = NumeralStyle.copy(fontSize = 14.sp),
                        color = Fresh,
                    )
                    Text(
                        text = " kept",
                        style = Typography.bodySmall,
                        color = Fresh,
                    )
                }
            }
        }
        Spacer(modifier = Modifier.width(12.dp))
        Column(horizontalAlignment = Alignment.End) {
            if (entry.purchased) {
                Text(
                    text = "₱%,.2f".format(entry.totalSpent),
                    style = NumeralStyle.copy(fontSize = 16.sp),
                    color = Ink,
                )
                Spacer(modifier = Modifier.height(4.dp))
            }
            if (upcoming && entry.saved && !entry.purchased) {
                Text(
                    text = "Unsave",
                    style = Typography.labelMedium,
                    color = ErrorRed,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.clickable { onUnsave() }
                )
            } else if (openable) {
                ArrowLabel(
                    text = "Open →",
                    color = Ink,
                    style = Typography.labelMedium,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }
    }
}
