package com.quickpitik.mobile.ui.runner

import android.content.ContentValues
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.*
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.*
import com.quickpitik.mobile.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
// Selfie cap, mirroring the website's SELFIE_MAX. Backend enforces the same
// limit; this drives the copy and hides the add tile once reached.
const val SELFIE_MAX = 5

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProfileScreen(
    viewModel: ProfileViewModel,
    cartViewModel: CartViewModel,
    savedEventsViewModel: SavedEventsViewModel,
    onOpenEvent: (String) -> Unit = {},
    onBrowseEvents: () -> Unit = {},
    onLogout: () -> Unit = {}
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

    // Trigger fetches if not loaded. Saved events are RUNNER-role-gated —
    // skipped for a photographer browsing in runner view (their race log then
    // shows purchases only, which for a photographer is none).
    val isTrueRunner = rememberIsTrueRunner()
    LaunchedEffect(Unit) {
        cartViewModel.fetchOrders()
        if (isTrueRunner) savedEventsViewModel.refresh()
        viewModel.fetchSelfies()
    }

    val orders = (ordersState as? OrdersState.Success)?.orders ?: emptyList()
    val raceLog = remember(orders, savedEvents) { buildRaceLog(savedEvents, orders) }
    val ordersLoading = ordersState is OrdersState.Loading

    val context = LocalContext.current
    var tempImageUri by remember { mutableStateOf<Uri?>(null) }
    // Declared before the launchers below, which write to it from callbacks.
    var snackbarMessage by remember { mutableStateOf<String?>(null) }

    // Multi-select (web selfie-library parity): pick several angles at once,
    // capped to the open slots. Overflow is announced, not silently dropped.
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetMultipleContents()
    ) { uris ->
        if (uris.isEmpty()) return@rememberLauncherForActivityResult
        val remaining = (SELFIE_MAX - selfies.size).coerceAtLeast(0)
        uris.take(remaining).forEach { viewModel.uploadSelfie(it) }
        if (uris.size > remaining) {
            snackbarMessage = "The library holds $SELFIE_MAX selfies — " +
                "${uris.size - remaining} didn't fit."
        }
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
    LaunchedEffect(snackbarMessage) {
        snackbarMessage?.let { msg ->
            snackbarHostState.showSnackbar(msg)
            snackbarMessage = null
        }
    }
    // Saved-events feedback (race-log Unsave) — with an Undo action, web
    // parity. This screen previously never surfaced the VM's message at all,
    // so an unsave here was completely silent.
    val savedMessage by savedEventsViewModel.message.collectAsState()
    LaunchedEffect(savedMessage) {
        savedMessage?.let { msg ->
            val undo = savedEventsViewModel.undoCandidate.value
            val result = snackbarHostState.showSnackbar(
                message = msg,
                actionLabel = if (undo != null) "Undo" else null,
            )
            if (result == SnackbarResult.ActionPerformed && undo != null) {
                savedEventsViewModel.undoUnsave(undo.first, undo.second)
            }
            savedEventsViewModel.clearMessage()
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Bone)) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(top = 24.dp)
        ) {
            // Top Bar
            RunnerTopBar(
                kicker = "RUNNER PROFILE",
                onLogout = onLogout
            )

            Spacer(modifier = Modifier.height(24.dp))

            // Pull-to-refresh (Mobile Design skill) — re-pulls selfies +
            // orders (+ saved events for true runners); spinner settles when
            // the slow fetch (orders) completes.
            var profileRefreshing by remember { mutableStateOf(false) }
            val refreshScope = rememberCoroutineScope()
            PullToRefreshBox(
                isRefreshing = profileRefreshing,
                onRefresh = {
                    profileRefreshing = true
                    refreshScope.launch {
                        viewModel.fetchSelfies()
                        if (isTrueRunner) savedEventsViewModel.refresh()
                        cartViewModel.fetchOrders().join()
                        profileRefreshing = false
                    }
                },
                modifier = Modifier.fillMaxSize(),
            ) {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 24.dp),
                verticalArrangement = Arrangement.spacedBy(24.dp),
                contentPadding = PaddingValues(bottom = 24.dp)
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
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Column {
                                Kicker("01 · Selfie library")
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    // Header line tracks primary-state, as on web:
                                    // once a primary exists, searches are live.
                                    text = if (selfies.any { it.isPrimary }) {
                                        "Searches running across every event you join."
                                    } else {
                                        "Pick a primary selfie below."
                                    },
                                    style = Typography.bodySmall,
                                    color = Slate
                                )
                            }
                            Row(
                                modifier = Modifier.fillMaxWidth(),
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
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text(
                                        text = "Build your selfie library.",
                                        color = Ink,
                                        textAlign = TextAlign.Center,
                                        style = Typography.titleMedium,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Spacer(modifier = Modifier.height(6.dp))
                                    Text(
                                        text = "Upload up to $SELFIE_MAX clear, frontal selfies so we can " +
                                            "match you across every event you join.",
                                        color = Slate,
                                        textAlign = TextAlign.Center,
                                        style = Typography.bodyMedium
                                    )
                                }
                            }
                        } else {
                            Spacer(modifier = Modifier.height(16.dp))
                            SelfieGrid(
                                selfies = selfies,
                                onDelete = { viewModel.deleteSelfie(it) },
                                onSetPrimary = { id ->
                                    hapticFire(QpHaptic.CONFIRM)
                                    viewModel.setPrimarySelfie(id)
                                    snackbarMessage = "Primary selfie updated."
                                },
                                onAdd = { galleryLauncher.launch("image/*") },
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = "${selfies.size} / $SELFIE_MAX selfies stored.",
                                style = Typography.bodySmall,
                                color = Slate,
                            )
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = "Reused across every event you join — no re-uploading per race.",
                                style = Typography.bodySmall,
                                color = SlateSoft,
                            )
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
                } else if (ordersState is OrdersState.Error && raceLog.isEmpty()) {
                    // A failed orders fetch used to render as "No races yet."
                    // — an error must never masquerade as an empty state.
                    item {
                        ErrorView(
                            message = (ordersState as OrdersState.Error).message,
                            title = "Couldn't load your races",
                            onRetry = { cartViewModel.fetchOrders() },
                            modifier = Modifier.fillMaxWidth(),
                        )
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

// Responsive selfie grid — port of the web's
// `grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 md:gap-4`. A manual
// chunked-Row grid rather than LazyVerticalGrid: this sits inside the profile's
// LazyColumn, and nesting a lazy scroller in the same axis crashes Compose.
// Bounded at SELFIE_MAX + 1 cells, so a plain grid is the right tool anyway.
@Composable
private fun SelfieGrid(
    selfies: List<SelfieRefDto>,
    onDelete: (String) -> Unit,
    onSetPrimary: (String) -> Unit,
    onAdd: () -> Unit,
) {
    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val columns = when {
            maxWidth >= 1024.dp -> 5
            maxWidth >= 640.dp -> 3
            else -> 2
        }
        val gap = if (maxWidth >= 768.dp) 16.dp else 12.dp
        // The trailing tile is the add affordance under the cap, and a cap
        // EXPLAINER at 5/5 (web SelfieCapTile parity — the tile silently
        // vanishing read as a bug, not a limit).
        val cellCount = selfies.size + 1

        Column(verticalArrangement = Arrangement.spacedBy(gap)) {
            (0 until cellCount step columns).forEach { rowStart ->
                Row(horizontalArrangement = Arrangement.spacedBy(gap)) {
                    (rowStart until rowStart + columns).forEach { cell ->
                        when {
                            cell < selfies.size -> {
                                val selfie = selfies[cell]
                                key(selfie.id) {
                                    SelfieCard(
                                        selfie = selfie,
                                        onDelete = { onDelete(selfie.id) },
                                        onSetPrimary = { onSetPrimary(selfie.id) },
                                        modifier = Modifier.weight(1f),
                                    )
                                }
                            }
                            cell < cellCount ->
                                if (selfies.size < SELFIE_MAX) SelfieAddTile(
                                    onClick = onAdd,
                                    modifier = Modifier.weight(1f),
                                ) else SelfieCapTile(modifier = Modifier.weight(1f))
                            // Empty filler keeps a partial row's tiles the same
                            // width as a full row's.
                            else -> Spacer(modifier = Modifier.weight(1f))
                        }
                    }
                }
            }
        }
    }
}

// Shown at 5/5 in place of the add tile — says WHY adding is closed (web
// SelfieCapTile parity).
@Composable
private fun SelfieCapTile(modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .aspectRatio(1f)
            .border(1.dp, Line, TileShape)
            .padding(8.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = "Library full — remove one to add another.",
            style = Typography.bodySmall,
            color = SlateSoft,
            textAlign = TextAlign.Center,
        )
    }
}

// Dashed "+" cell for adding a selfie from the gallery. Live camera capture
// stays on the header button, matching the existing two-affordance split.
@Composable
private fun SelfieAddTile(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .aspectRatio(1f)
            .clip(QpCardShape)
            .drawBehind {
                drawRoundRect(
                    color = Line,
                    style = Stroke(
                        width = 1.dp.toPx(),
                        pathEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 8f)),
                    ),
                    cornerRadius = CornerRadius(16.dp.toPx()),
                )
            }
            .clickable { onClick() },
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(text = "+", color = Slate, style = Typography.titleLarge)
        Spacer(modifier = Modifier.height(4.dp))
        Kicker(text = "Add selfie", color = SlateSoft)
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
            // Square, matching web's `aspect-square`.
            .aspectRatio(1f)
            .clip(QpCardShape)
            .background(BoneDeep)
            .clickable { if (!selfie.isPrimary) onSetPrimary() }
    ) {
        AsyncImage(
            model = RetrofitClient.resolveImageUrl(selfie.dataUrl),
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
                // Quality badge — slate scrim, not pure black. Reads
                // qualityTestStatus, not qualityScore: with AI_API_ENABLED=false
                // the gate is skipped and every score is 0, so the old
                // "Q 0%" label was meaningless on every tile. "Not checked"
                // says the true thing — this selfie has never been tested.
                val passed = selfie.qualityTestStatus == "passed"
                Box(
                    modifier = Modifier
                        .background(Ink.copy(alpha = 0.7f), BadgeShape)
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = if (passed) {
                            "Q ${(selfie.qualityScore * 100).toInt()}%"
                        } else {
                            "Not checked"
                        },
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
