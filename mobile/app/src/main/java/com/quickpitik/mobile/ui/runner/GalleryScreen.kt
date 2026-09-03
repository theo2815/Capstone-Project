package com.quickpitik.mobile.ui.runner

import android.content.ContentValues
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Place
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.material3.BottomSheetDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.foundation.selection.toggleable
import androidx.compose.ui.semantics.Role
import androidx.compose.material3.IconButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
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
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.ViewMode
import com.quickpitik.mobile.data.local.isPhotographerRole
import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.QpWebSocket
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.WsState
import com.quickpitik.mobile.ui.theme.ArrowLabel
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.BrandLogo
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.ErrorView
import com.quickpitik.mobile.ui.theme.FieldShape
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.LoadingSkeleton
import com.quickpitik.mobile.ui.theme.NumeralStyle
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.StatusChip
import com.quickpitik.mobile.ui.theme.StatusTone
import com.quickpitik.mobile.ui.theme.TileShape
import com.quickpitik.mobile.ui.theme.Typography
import com.quickpitik.mobile.ui.theme.SecureScreen
import kotlinx.coroutines.launch
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RunnerGalleryScreen(
    viewModel: RunnerGalleryViewModel,
    cartViewModel: CartViewModel,
    inboxViewModel: RunnerInboxViewModel,
    savedEventsViewModel: SavedEventsViewModel,
    onNavigateToProfile: () -> Unit,
    onNavigateBack: () -> Unit,
    onOpenOrder: (String) -> Unit,
    onOpenPhotographer: (String) -> Unit,
    onLogout: () -> Unit
) {
    // Runner browse surface — every photo here is an unpurchased preview, so
    // screenshots/recording/casting are blocked for the whole screen.
    SecureScreen()

    // rememberSaveable: the typed bib + chosen search tab survive rotation.
    var bibSearchQuery by rememberSaveable { mutableStateOf("") }
    var activeSearchTab by rememberSaveable { mutableStateOf(0) } // 0 = Selfie, 1 = Bib Number
    var selectedPhotoForDetail by remember { mutableStateOf<PhotoDto?>(null) }

    val inboxMessages by inboxViewModel.messages.collectAsState()
    val inboxUnread by inboxViewModel.unreadCount.collectAsState()
    var showInbox by remember { mutableStateOf(false) }
    var showRefundPolicy by remember { mutableStateOf(false) }
    var showEventInfoSheet by rememberSaveable { mutableStateOf(false) }

    val activeEvent by viewModel.activeEvent.collectAsState()
    val searchState by viewModel.searchState.collectAsState()
    val isFiltered by viewModel.isFiltered.collectAsState()
    val newPhotoCount by viewModel.newPhotoCount.collectAsState()
    val liveState by viewModel.liveState.collectAsState()
    val photoAlert by viewModel.photoAlert.collectAsState()
    // Hoisted so the live-photos pill can jump the runner back to the top when
    // new shots land while they're scrolled down the grid.
    val gridState = rememberLazyGridState()
    val scope = rememberCoroutineScope()

    // Push channels are held only while the cockpit is actually on screen. A
    // race lasts hours; a socket surviving in a pocketed phone would burn
    // battery for frames nobody can see, and Android freezes cached processes
    // anyway. Every reconnect refetches, so nothing is missed on return.
    // RUNNER-role-gated affordances (cart, photo alerts, runner inbox) are
    // hidden for a photographer browsing in runner view (ViewMode) — the
    // backend 403s a photographer token on those endpoints. Browsing, search,
    // the lightbox, and live photos all stay live for both.
    val isTrueRunner = rememberIsTrueRunner()
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner, activeEvent?.id) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> {
                    viewModel.connectLivePhotos()
                    if (isTrueRunner) inboxViewModel.connect()
                }
                Lifecycle.Event.ON_STOP -> {
                    viewModel.disconnectLivePhotos()
                    if (isTrueRunner) inboxViewModel.disconnect()
                }
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            viewModel.disconnectLivePhotos()
            if (isTrueRunner) inboxViewModel.disconnect()
        }
    }
    // Hoisted so the grid tile's inline +cart / buy buttons can read in-cart
    // state for the ✓-cart label flip without each tile collecting its own copy.
    val cartItems by cartViewModel.cartItems.collectAsState()

    // Live selfie capture (camera) + gallery pick — reuse the proven ProfileScreen
    // pattern: a MediaStore URI handed to TakePicture(), then face-search the bytes.
    val context = LocalContext.current
    var pendingSelfieUri by remember { mutableStateOf<Uri?>(null) }
    val selfieCameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) pendingSelfieUri?.let { viewModel.searchBySelfieUri(it) }
    }
    val selfieGalleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { viewModel.searchBySelfieUri(it) }
    }

    // One selfie-library request feeds both the picker and the runner-only
    // photo-alert check; these used to request the same list concurrently.
    LaunchedEffect(activeEvent?.slug, isTrueRunner) {
        viewModel.loadGalleryMetadata(activeEvent?.slug.takeIf { isTrueRunner })
    }
    val eventDetail by viewModel.eventDetail.collectAsState()

    // Lock the Runner Dashboard to the uniform Light Warm Cream Brand Theme
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Bone
    ) {
        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            state = gridState,
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding(),
            contentPadding = PaddingValues(24.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Header Row
            item(span = { GridItemSpan(maxLineSpan) }) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        BrandLogo(compact = true)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "GALLERY HUB",
                            style = Typography.labelMedium,
                            color = Slate
                        )
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        // Save-event affordance ON the event page (web parity:
                        // the cockpit header carries a SaveButton) — previously
                        // save only existed back on the discovery tile.
                        if (isTrueRunner) {
                            val savedIds by savedEventsViewModel.savedIds.collectAsState()
                            val event = activeEvent
                            if (event != null) {
                                val saved = event.id in savedIds
                                IconButton(onClick = { savedEventsViewModel.toggle(event) }) {
                                    Icon(
                                        imageVector = if (saved) Icons.Default.Favorite
                                        else Icons.Default.FavoriteBorder,
                                        contentDescription = if (saved) "Remove from saved" else "Save event",
                                        tint = if (saved) Fresh else Slate,
                                    )
                                }
                            }
                            RunnerInboxBell(
                                messageCount = inboxMessages.size,
                                unreadCount = inboxUnread,
                                onClick = { showInbox = true },
                            )
                        }
                        // Cart access lives in the global FloatingCart pill — header icon dropped
                        // to avoid two affordances pointing at the same overlay.
                        var menuExpanded by remember { mutableStateOf(false) }
                        val sessionManager = remember { SessionManager.getInstance(context) }
                        val userName = sessionManager.getUserName() ?: "Runner"

                        Box {
                            Box(
                                modifier = Modifier
                                    .size(40.dp)
                                    .clip(CircleShape)
                                    .background(BoneDeep)
                                    .clickable { menuExpanded = true },
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = userName.take(1).uppercase(),
                                    color = Ink,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 16.sp
                                )
                            }

                            DropdownMenu(
                                expanded = menuExpanded,
                                onDismissRequest = { menuExpanded = false },
                                modifier = Modifier.background(Bone)
                            ) {
                                // Same items + labels as RunnerTopBar's menu —
                                // this inline copy had drifted ("Sign Out").
                                if (isPhotographerRole(sessionManager.getUserRole())) {
                                    DropdownMenuItem(
                                        text = { Text("Switch to photographer", color = Ink) },
                                        onClick = {
                                            menuExpanded = false
                                            ViewMode.requestSwitchToPhotographer()
                                        }
                                    )
                                }
                                DropdownMenuItem(
                                    text = { Text("Log out", color = ErrorRed) },
                                    onClick = {
                                        menuExpanded = false
                                        onLogout()
                                    }
                                )
                            }
                        }
                    }
                }
            }

            if (activeEvent == null) {
                item(span = { GridItemSpan(maxLineSpan) }) {
                    LoadingSkeleton(
                        shape = QpCardShape,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(220.dp),
                    )
                }
            } else if (deriveEventState(activeEvent!!.date) == EventState.UPCOMING) {
                // Pre-race-day: no gallery, no search. Port of the website's
                // UpcomingEventNotice branch in events/[slug]/page.tsx.
                item(span = { GridItemSpan(maxLineSpan) }) {
                    UpcomingEventNotice(
                        event = activeEvent!!,
                        onBack = {
                            viewModel.clearSelectedEvent()
                            onNavigateBack()
                        },
                    )
                }
                // Pre-event opt-in — "notify me when my photos are ready" is most
                // useful before race day, when there's nothing to search yet.
                // RUNNER-gated — hidden in photographer runner-view.
                if (isTrueRunner) item(span = { GridItemSpan(maxLineSpan) }) {
                    PhotoAlertCard(
                        state = photoAlert,
                        onToggle = { register ->
                            activeEvent?.slug?.let { viewModel.togglePhotoAlert(it, register) }
                        },
                        onAddSelfie = onNavigateToProfile,
                    )
                }
            } else {
                // SELECTED EVENT GALLERY VIEW
                // Compact Race Identity Strip — keeps photos front and center on load!
                item(span = { GridItemSpan(maxLineSpan) }) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(BoneDeep, QpCardShape)
                            .border(1.dp, Line, QpCardShape)
                            .clickable { showEventInfoSheet = true }
                            .padding(horizontal = 14.dp, vertical = 12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f).padding(end = 8.dp)) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier
                                    .clip(PillShape)
                                    .clickable {
                                        viewModel.clearSelectedEvent()
                                        onNavigateBack()
                                    }
                                    .padding(vertical = 2.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.ArrowBack,
                                    contentDescription = "Back to Events",
                                    tint = Slate,
                                    modifier = Modifier.size(14.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text(
                                    text = "ALL RACES",
                                    style = Typography.labelSmall.copy(fontWeight = FontWeight.Bold, letterSpacing = 0.5.sp),
                                    color = Slate
                                )
                            }
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = activeEvent?.name.orEmpty(),
                                style = Typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = Ink,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                activeEvent?.date?.let { dateStr ->
                                    Text(
                                        text = eventDateLabel(dateStr),
                                        style = Typography.bodySmall,
                                        color = Slate
                                    )
                                }
                                if (!activeEvent?.location.isNullOrBlank()) {
                                    Text(text = "·", color = SlateSoft, style = Typography.bodySmall)
                                    Text(
                                        text = extractCity(activeEvent?.location.orEmpty()),
                                        style = Typography.bodySmall,
                                        color = Slate,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }
                            }
                        }

                        // Right: Status Chip + Info pill
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            activeEvent?.let { ev ->
                                StatusChip(state = deriveEventState(ev.date))
                            }
                            Surface(
                                shape = PillShape,
                                color = Bone,
                                border = BorderStroke(1.dp, Line),
                                modifier = Modifier
                                    .clip(PillShape)
                                    .clickable { showEventInfoSheet = true }
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Info,
                                        contentDescription = "Race Info",
                                        tint = Slate,
                                        modifier = Modifier.size(15.dp)
                                    )
                                    Text(
                                        text = "Race info",
                                        style = Typography.labelSmall.copy(fontWeight = FontWeight.SemiBold, letterSpacing = 0.2.sp),
                                        color = Ink
                                    )
                                }
                            }
                        }
                    }
                }

                // AI Search Mode Selector (Segmented Pill: Face vs Bib)
                item(span = { GridItemSpan(maxLineSpan) }) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(BoneDeep, PillShape)
                            .border(1.dp, Line, PillShape)
                            .padding(4.dp),
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        // AI Face Search Tab
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .clip(PillShape)
                                .background(if (activeSearchTab == 0) Ink else Color.Transparent)
                                .clickable { activeSearchTab = 0 }
                                .padding(vertical = 10.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Face,
                                    contentDescription = null,
                                    tint = if (activeSearchTab == 0) Fresh else Slate,
                                    modifier = Modifier.size(18.dp)
                                )
                                Text(
                                    text = "AI Face Search",
                                    style = Typography.labelMedium,
                                    fontWeight = if (activeSearchTab == 0) FontWeight.Bold else FontWeight.Medium,
                                    color = if (activeSearchTab == 0) Bone else Slate
                                )
                            }
                        }

                        // Bib Lookup Tab
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .clip(PillShape)
                                .background(if (activeSearchTab == 1) Ink else Color.Transparent)
                                .clickable { activeSearchTab = 1 }
                                .padding(vertical = 10.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Search,
                                    contentDescription = null,
                                    tint = if (activeSearchTab == 1) Fresh else Slate,
                                    modifier = Modifier.size(18.dp)
                                )
                                Text(
                                    text = "Bib Number",
                                    style = Typography.labelMedium,
                                    fontWeight = if (activeSearchTab == 1) FontWeight.Bold else FontWeight.Medium,
                                    color = if (activeSearchTab == 1) Bone else Slate
                                )
                            }
                        }
                    }
                }

                // Search Action Interface
                item(span = { GridItemSpan(maxLineSpan) }) {
                    if (activeSearchTab == 0) {
                        // Selfie Upload Action Trigger
                        Card(
                            border = BorderStroke(1.dp, Line),
                            colors = CardDefaults.cardColors(containerColor = BoneDeep),
                            shape = QpCardShape,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(
                                modifier = Modifier.padding(20.dp),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Text(
                                    text = "Find photos of yourself instantly with face recognition.",
                                    textAlign = TextAlign.Center,
                                    style = Typography.bodyMedium,
                                    color = InkSoft
                                )
                                Spacer(modifier = Modifier.height(12.dp))
                                // Selfie picker — web SelfieSearchPanel parity:
                                // the runner picks WHICH stored selfie to match
                                // with; tapping a thumbnail fires the search.
                                // Falls back to the primary-selfie CTA when the
                                // library hasn't loaded (or is empty).
                                val librarySelfies by viewModel.selfies.collectAsState()
                                if (librarySelfies.isNotEmpty()) {
                                    // Whole-library match is the default action: every
                                    // saved selfie is another angle, and the backend
                                    // unions the matches (2026-09-02).
                                    PrimaryCta(
                                        text = "Search with all my selfies (${librarySelfies.size})",
                                        onClick = { viewModel.searchWithAllSelfies() },
                                        modifier = Modifier.fillMaxWidth(),
                                    )
                                    Spacer(modifier = Modifier.height(10.dp))
                                    Row(
                                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                                        modifier = Modifier.fillMaxWidth(),
                                    ) {
                                        librarySelfies.take(5).forEach { selfie ->
                                            Box(
                                                modifier = Modifier
                                                    .weight(1f)
                                                    .aspectRatio(1f)
                                                    .clip(TileShape)
                                                    .border(
                                                        width = if (selfie.isPrimary) 2.dp else 1.dp,
                                                        color = if (selfie.isPrimary) Fresh else Line,
                                                        shape = TileShape,
                                                    )
                                                    .clickable { viewModel.searchBySelfieId(selfie.id) },
                                            ) {
                                                AsyncImage(
                                                    model = RetrofitClient.resolveImageUrl(selfie.dataUrl),
                                                    contentDescription = "Search with this selfie",
                                                    contentScale = ContentScale.Crop,
                                                    modifier = Modifier.fillMaxSize(),
                                                )
                                                if (selfie.isPrimary) {
                                                    Kicker(
                                                        text = "Primary",
                                                        color = Fresh,
                                                        modifier = Modifier
                                                            .align(Alignment.BottomCenter)
                                                            .background(Bone.copy(alpha = 0.85f))
                                                            .padding(horizontal = 4.dp, vertical = 1.dp),
                                                    )
                                                }
                                            }
                                        }
                                    }
                                    Spacer(modifier = Modifier.height(6.dp))
                                    Text(
                                        text = "Or tap one selfie to search with just that one.",
                                        style = Typography.bodySmall,
                                        color = SlateSoft,
                                        textAlign = TextAlign.Center,
                                        modifier = Modifier.fillMaxWidth(),
                                    )
                                } else {
                                    // Primary: search with a saved library selfie
                                    PrimaryCta(
                                        text = "Search with stored selfie",
                                        onClick = { viewModel.searchByStoredSelfie() },
                                        modifier = Modifier.fillMaxWidth(),
                                    )
                                }
                                Spacer(modifier = Modifier.height(8.dp))
                                Row(
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    // Take a selfie now — opens the camera via a MediaStore URI
                                    GhostCta(
                                        text = "Take selfie",
                                        onClick = {
                                            try {
                                                val values = ContentValues().apply {
                                                    put(MediaStore.Images.Media.TITLE, "selfie_search_${System.currentTimeMillis()}")
                                                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                                                }
                                                val uri = context.contentResolver.insert(
                                                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                                                    values
                                                )
                                                pendingSelfieUri = uri
                                                if (uri != null) selfieCameraLauncher.launch(uri)
                                            } catch (e: Exception) {
                                                android.widget.Toast.makeText(context, "Unable to open camera.", android.widget.Toast.LENGTH_SHORT).show()
                                            }
                                        },
                                        modifier = Modifier.weight(1f),
                                    )
                                    // Upload an existing photo from the device
                                    GhostCta(
                                        text = "Upload",
                                        onClick = { selfieGalleryLauncher.launch("image/*") },
                                        modifier = Modifier.weight(1f),
                                    )
                                }
                                // Save-and-search (2026-09-02): keep the new selfie
                                // without a trip to Profile. Hidden when the library
                                // is full or the viewer isn't a runner (the endpoint
                                // is RUNNER-gated server-side).
                                val saveToLibrary by viewModel.saveSelfieToLibrary.collectAsState()
                                val saveNotice by viewModel.selfieSaveNotice.collectAsState()
                                if (isTrueRunner && librarySelfies.size < SELFIE_MAX) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .heightIn(min = 48.dp)
                                            .toggleable(
                                                value = saveToLibrary,
                                                role = Role.Checkbox,
                                                onValueChange = { viewModel.setSaveSelfieToLibrary(it) },
                                            ),
                                    ) {
                                        Checkbox(
                                            checked = saveToLibrary,
                                            onCheckedChange = null,
                                            colors = CheckboxDefaults.colors(checkedColor = Ink, checkmarkColor = Bone),
                                        )
                                        Text(
                                            text = "Also save it to my selfie library (${librarySelfies.size} of $SELFIE_MAX)",
                                            style = Typography.bodySmall,
                                            color = Slate,
                                        )
                                    }
                                }
                                saveNotice?.let { notice ->
                                    Text(
                                        text = notice,
                                        style = Typography.bodySmall,
                                        color = SlateSoft,
                                        textAlign = TextAlign.Center,
                                        modifier = Modifier.fillMaxWidth(),
                                    )
                                }
                            }
                        }
                    } else {
                        // Bib Entry Action Input
                        TextField(
                            value = bibSearchQuery,
                            onValueChange = {
                                bibSearchQuery = it
                                viewModel.searchByBib(it)
                            },
                            placeholder = { Text("Enter bib number (e.g. 2948)", color = SlateSoft) },
                            leadingIcon = {
                                Icon(
                                    imageVector = Icons.Default.Search,
                                    contentDescription = null,
                                    tint = if (bibSearchQuery.isNotEmpty()) Fresh else SlateSoft,
                                    modifier = Modifier.size(20.dp)
                                )
                            },
                            trailingIcon = if (bibSearchQuery.isNotEmpty()) {
                                {
                                    IconButton(
                                        onClick = {
                                            bibSearchQuery = ""
                                            viewModel.searchByBib("")
                                        }
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Close,
                                            contentDescription = "Clear search",
                                            tint = Slate,
                                            modifier = Modifier.size(18.dp)
                                        )
                                    }
                                }
                            } else null,
                            // Bibs are alphanumeric (web parity: text input, uppercased).
                            keyboardOptions = KeyboardOptions(
                                keyboardType = KeyboardType.Text,
                                capitalization = KeyboardCapitalization.Characters,
                            ),
                            singleLine = true,
                            colors = TextFieldDefaults.colors(
                                focusedContainerColor = BoneDeep,
                                unfocusedContainerColor = BoneDeep,
                                focusedIndicatorColor = Fresh,
                                unfocusedIndicatorColor = Color.Transparent,
                                focusedTextColor = Ink,
                                unfocusedTextColor = InkSoft
                            ),
                            shape = FieldShape,
                            modifier = Modifier
                                .fillMaxWidth()
                                .border(1.dp, Line, FieldShape)
                        )
                    }
                }

                // Watermarked Photo Stream Title with Reset Button
                item(span = { GridItemSpan(maxLineSpan) }) {
                    val photoCount = (searchState as? PhotosSearchState.Success)?.photos?.size
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Kicker(if (isFiltered) "MATCHED PHOTOS" else "PHOTO STREAM")
                                if (photoCount != null && photoCount > 0) {
                                    Box(
                                        modifier = Modifier
                                            .clip(PillShape)
                                            .background(if (isFiltered) Fresh.copy(alpha = 0.15f) else Line)
                                            .padding(horizontal = 8.dp, vertical = 2.dp)
                                    ) {
                                        Text(
                                            text = "$photoCount",
                                            style = NumeralStyle.copy(fontSize = 12.sp, fontWeight = FontWeight.Bold),
                                            color = if (isFiltered) Fresh else Ink
                                        )
                                    }
                                }
                            }
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = if (isFiltered) "Photos matched to your search" else "Watermarked previews · buy to unlock full-res",
                                style = Typography.bodySmall,
                                color = SlateSoft,
                            )
                        }
                        if (isFiltered || searchState is PhotosSearchState.Error) {
                            Surface(
                                shape = PillShape,
                                color = BoneDeep,
                                border = BorderStroke(1.dp, Line),
                                modifier = Modifier.clickable {
                                    bibSearchQuery = ""
                                    viewModel.clearFilter()
                                }
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Refresh,
                                        contentDescription = null,
                                        tint = Slate,
                                        modifier = Modifier.size(14.dp)
                                    )
                                    Text(
                                        text = "Reset",
                                        style = Typography.labelSmall,
                                        color = Ink,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                }
                            }
                        } else if (isTrueRunner) {
                            Surface(
                                shape = PillShape,
                                color = BoneDeep,
                                border = BorderStroke(1.dp, Line),
                                modifier = Modifier.clickable {
                                    bibSearchQuery = ""
                                    viewModel.searchByStoredSelfie()
                                }
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Face,
                                        contentDescription = null,
                                        tint = Fresh,
                                        modifier = Modifier.size(14.dp)
                                    )
                                    Text(
                                        text = "My photos",
                                        style = Typography.labelSmall,
                                        color = Ink,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                }
                            }
                        }
                    }
                }

                item(span = { GridItemSpan(maxLineSpan) }) {
                    LivePhotosBanner(
                        newPhotoCount = newPhotoCount,
                        liveState = liveState,
                        onJumpToTop = {
                            viewModel.refreshLivePhotos()
                            scope.launch { gridState.animateScrollToItem(0) }
                        },
                        onRetry = { viewModel.retryLivePhotos() },
                    )
                }

                // Beautiful Watermarked Photo Grid
                when (val state = searchState) {
                    is PhotosSearchState.Loading -> {
                        item(span = { GridItemSpan(maxLineSpan) }) {
                            PhotoGridSkeleton()
                        }
                    }
                    is PhotosSearchState.Error -> {
                        item(span = { GridItemSpan(maxLineSpan) }) {
                            // Retry re-runs the typed bib query (or the plain
                            // browse when empty) — for a failed FACE search
                            // this lands back on the browsable wall, the same
                            // recovery the website's FaceEmptyResult offers.
                            ErrorView(
                                message = state.message,
                                onRetry = { viewModel.searchByBib(bibSearchQuery) },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                    }
                    is PhotosSearchState.Success -> {
                        if (state.photos.isEmpty()) {
                            item(span = { GridItemSpan(maxLineSpan) }) {
                                // Status-aware empty copy — ports the website's
                                // BibEmptyResult + FaceEmptyResult in place of
                                // the old one-size-fits-all string. (The web's
                                // notify-me email form is deliberately absent:
                                // the PhotoAlertCard above is mobile's native
                                // equivalent of that intent.)
                                val trimmedBib = bibSearchQuery.trim()
                                val eventState = activeEvent?.let { deriveEventState(it.date) }
                                val isFaceEmpty = isFiltered && trimmedBib.isEmpty()
                                val (emptyTitle, emptyBody) = when {
                                    isFaceEmpty ->
                                        "We didn't find your face." to
                                            "Try adding another selfie angle, or browse the wall while photos roll in."
                                    isFiltered && eventState == EventState.LIVE ->
                                        "Still uploading." to
                                            "Photographers are still working through this race — check back soon for $trimmedBib."
                                    isFiltered && eventState == EventState.PAST ->
                                        "This race has wrapped." to
                                            "Photos for $trimmedBib never landed in this archive. The wall's still here if you want to skim."
                                    isFiltered ->
                                        "Bib not found." to
                                            "All photos for this race have been uploaded — $trimmedBib isn't in there. Double-check the number, or skim the wall."
                                    else ->
                                        "No photos yet." to
                                            "Nothing has been uploaded for this event so far."
                                }
                                Column(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(vertical = 32.dp),
                                    horizontalAlignment = Alignment.CenterHorizontally,
                                ) {
                                    Text(
                                        text = emptyTitle,
                                        color = Ink,
                                        textAlign = TextAlign.Center,
                                        style = Typography.titleMedium,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Spacer(modifier = Modifier.height(6.dp))
                                    Text(
                                        text = emptyBody,
                                        color = SlateSoft,
                                        textAlign = TextAlign.Center,
                                        style = Typography.bodyMedium,
                                    )
                                    if (isFiltered) {
                                        Spacer(modifier = Modifier.height(16.dp))
                                        GhostCta(
                                            text = "Browse the wall →",
                                            onClick = {
                                                bibSearchQuery = ""
                                                viewModel.clearFilter()
                                            },
                                        )
                                    }
                                }
                            }
                        } else {
                            // "Buy all N · ₱X →" — web BuyAllBar parity: one
                            // tap adds every visible match to the cart. Shown
                            // only on a filtered result set (your matches, not
                            // the whole wall) and only for true runners (cart
                            // is RUNNER-gated server-side).
                            if (isFiltered && isTrueRunner) {
                                item(span = { GridItemSpan(maxLineSpan) }) {
                                    val allInCart = state.photos.all { p ->
                                        cartItems.any { it.photoId == p.id }
                                    }
                                    val bulkTotal = state.photos.sumOf { it.price }
                                    PrimaryCta(
                                        text = when {
                                            allInCart -> "Added · ${state.photos.size} in cart ✓"
                                            state.photos.size == 1 ->
                                                "Buy 1 · ₱${"%,.0f".format(bulkTotal)} →"
                                            else ->
                                                "Buy all ${state.photos.size} · ₱${"%,.0f".format(bulkTotal)} →"
                                        },
                                        enabled = !allInCart,
                                        onClick = {
                                            val event = activeEvent ?: return@PrimaryCta
                                            state.photos.forEach { p ->
                                                if (cartItems.none { it.photoId == p.id }) {
                                                    cartViewModel.addToCart(
                                                        p, event.id, event.slug, event.name,
                                                    )
                                                }
                                            }
                                        },
                                        modifier = Modifier.fillMaxWidth(),
                                    )
                                }
                            }
                            items(state.photos, key = { it.id }) { photo ->
                                val photoInCart = cartItems.any { it.photoId == photo.id }
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .aspectRatio(0.85f)
                                        .clip(QpCardShape)
                                        .background(BoneDeep)
                                        .clickable { selectedPhotoForDetail = photo },
                                    contentAlignment = Alignment.Center
                                ) {
                                    if (photo.imageUrl != null) {
                                        AsyncImage(
                                            model = RetrofitClient.resolveImageUrl(photo.imageUrl),
                                            contentDescription = "Photo preview",
                                            modifier = Modifier.fillMaxSize(),
                                            contentScale = ContentScale.Crop
                                        )
                                    }
                                    // No client-side scrim or "PREVIEW" text:
                                    // the backend bakes the QuickPitik credit
                                    // + photographer logo into imageUrl.

                                    // Inline cart / buy actions — bottom-right of tile.
                                    Row(
                                        modifier = Modifier
                                            .align(Alignment.BottomEnd)
                                            .padding(8.dp),
                                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                                    ) {
                                        // Cart is RUNNER-gated server-side —
                                        // pills hidden in photographer runner-view.
                                        if (isTrueRunner) {
                                            TileActionPill(
                                                label = if (photoInCart) "✓ cart" else "+ cart",
                                                filled = photoInCart,
                                                onClick = {
                                                    val event = activeEvent ?: return@TileActionPill
                                                    if (photoInCart) {
                                                        cartViewModel.removeFromCart(photo.id)
                                                    } else {
                                                        cartViewModel.addToCart(
                                                            photo, event.id, event.slug, event.name,
                                                        )
                                                    }
                                                },
                                            )
                                            TileActionPill(
                                                label = "buy →",
                                                filled = false,
                                                onClick = {
                                                    val event = activeEvent ?: return@TileActionPill
                                                    if (photoInCart) {
                                                        cartViewModel.openCheckoutSheet()
                                                    } else {
                                                        cartViewModel.triggerExpressCheckout()
                                                        cartViewModel.addToCart(
                                                            photo, event.id, event.slug, event.name,
                                                        )
                                                    }
                                                },
                                            )
                                        }
                                    }
                                }
                            }
                            // Load more + "Showing first N of M" — web
                            // LoadMoreButton parity. Face results are one-shot
                            // (total == size), so this renders only for the
                            // paged bib/browse queries.
                            if (state.total > state.photos.size) {
                                item(span = { GridItemSpan(maxLineSpan) }) {
                                    Column(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(vertical = 8.dp),
                                        horizontalAlignment = Alignment.CenterHorizontally,
                                    ) {
                                        GhostCta(
                                            text = if (state.loadingMore) "Loading…" else "Load more",
                                            enabled = !state.loadingMore,
                                            onClick = { viewModel.loadMorePhotos() },
                                            modifier = Modifier.fillMaxWidth(),
                                        )
                                        Spacer(modifier = Modifier.height(6.dp))
                                        Kicker(
                                            text = "Showing first ${state.photos.size} of ${state.total}",
                                            color = SlateSoft,
                                        )
                                    }
                                }
                            }
                        }
                    }
                    else -> {
                        item(span = { GridItemSpan(maxLineSpan) }) {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 32.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = "Please select a marathon and run a search to fetch premium matched photos.",
                                    color = SlateSoft,
                                    textAlign = TextAlign.Center,
                                    style = Typography.bodyMedium
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    val photoForDetail = selectedPhotoForDetail
    if (photoForDetail != null) {
        val photo = photoForDetail
        // Pull the whole matched-photos list so the lightbox pager can swipe
        // through every photo, not just the tapped one.
        val allPhotos = (searchState as? PhotosSearchState.Success)?.photos.orEmpty()
        val previewPhotos = allPhotos.map { it.toPreviewData(activeEvent?.name) }
        val currentIndex = previewPhotos.indexOfFirst { it.id == photo.id }

        if (currentIndex >= 0) {
            PhotoPreview(
                photos = previewPhotos,
                currentIndex = currentIndex,
                commerceEnabled = isTrueRunner,
                isInCart = { previewData ->
                    cartItems.any { it.photoId == previewData.id }
                },
                onClose = { selectedPhotoForDetail = null },
                onIndexChange = { newIndex ->
                    selectedPhotoForDetail = allPhotos.getOrNull(newIndex)
                },
                onToggleCart = { previewData ->
                    val targetPhoto = allPhotos.firstOrNull { it.id == previewData.id }
                        ?: return@PhotoPreview
                    val event = activeEvent
                    if (cartItems.any { it.photoId == previewData.id }) {
                        cartViewModel.removeFromCart(previewData.id)
                    } else if (event != null) {
                        cartViewModel.addToCart(targetPhoto, event.id, event.slug, event.name)
                    }
                    // Stay open so the runner sees the IN CART pill flip — they
                    // can dismiss the lightbox when they're done browsing.
                },
                onBuyNow = { previewData ->
                    val targetPhoto = allPhotos.firstOrNull { it.id == previewData.id }
                        ?: return@PhotoPreview
                    val event = activeEvent
                    if (cartItems.any { it.photoId == previewData.id }) {
                        cartViewModel.openCheckoutSheet()
                    } else if (event != null) {
                        cartViewModel.triggerExpressCheckout()
                        cartViewModel.addToCart(targetPhoto, event.id, event.slug, event.name)
                    }
                    selectedPhotoForDetail = null
                },
                onOpenPhotographer = { handle ->
                    // Close the lightbox first — otherwise device-back from the
                    // profile lands on a re-opened dialog instead of the grid.
                    selectedPhotoForDetail = null
                    onOpenPhotographer(handle)
                },
            )
        } else {
            // Photo no longer in the loaded set (e.g. search ran again under us).
            selectedPhotoForDetail = null
        }
    }

    if (showRefundPolicy) {
        RefundPolicyDialog(onDismiss = { showRefundPolicy = false })
    }

    if (showEventInfoSheet) {
        ModalBottomSheet(
            onDismissRequest = { showEventInfoSheet = false },
            containerColor = Bone,
            dragHandle = { BottomSheetDefaults.DragHandle(color = Line) },
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 24.dp)
                    .padding(bottom = 40.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // High-res event banner
                if (!activeEvent?.bannerUrl.isNullOrEmpty()) {
                    AsyncImage(
                        model = RetrofitClient.resolveImageUrl(activeEvent?.bannerUrl),
                        contentDescription = "Event Banner",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(180.dp)
                            .clip(QpCardShape),
                        contentScale = ContentScale.Crop
                    )
                }

                // Race Identity: Date, Name, Location & Status
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        activeEvent?.date?.let { dateStr ->
                            Kicker(eventDateLabel(dateStr))
                        }
                        activeEvent?.let { ev ->
                            StatusChip(state = deriveEventState(ev.date))
                        }
                    }
                    Text(
                        text = activeEvent?.name.orEmpty(),
                        style = Typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        color = Ink
                    )
                    if (!activeEvent?.location.isNullOrBlank()) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Place,
                                contentDescription = null,
                                tint = Fresh,
                                modifier = Modifier.size(16.dp)
                            )
                            Text(
                                text = activeEvent?.location.orEmpty(),
                                style = Typography.bodyMedium,
                                color = Slate
                            )
                        }
                    }
                }

                // About Details: Organizer, Description, Tags & Pricing
                eventDetail?.let { detail ->
                    QpCard(modifier = Modifier.fillMaxWidth()) {
                        Kicker("About this race", color = Slate)
                        if (!detail.organizerName.isNullOrBlank()) {
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Organized by ${detail.organizerName}",
                                style = Typography.titleSmall,
                                fontWeight = FontWeight.SemiBold,
                                color = Ink,
                            )
                        }
                        if (!detail.description.isNullOrBlank()) {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = detail.description,
                                style = Typography.bodyMedium,
                                color = Slate,
                            )
                        }
                        if (detail.categories.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(10.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                detail.categories.take(4).forEach { category ->
                                    StatusChip(text = category, tone = StatusTone.Neutral)
                                }
                            }
                        }
                        Spacer(modifier = Modifier.height(12.dp))
                        Divider(color = Line)
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(
                            text = buildString {
                                append("₱${"%,.0f".format(detail.pricePerPhoto)} per photo")
                                if (detail.bundlePrice != null && detail.bundleSize != null) {
                                    append(" · or ₱${"%,.0f".format(detail.bundlePrice)} for ${detail.bundleSize}")
                                }
                            },
                            style = NumeralStyle.copy(fontSize = 14.sp),
                            color = Ink,
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Watermarked previews are free. Pay once, download forever.",
                            style = Typography.bodySmall,
                            color = SlateSoft,
                        )
                    }
                }

                // Pre-event photo alert opt-in
                if (isTrueRunner) {
                    PhotoAlertCard(
                        state = photoAlert,
                        onToggle = { register ->
                            activeEvent?.slug?.let { viewModel.togglePhotoAlert(it, register) }
                        },
                        onAddSelfie = onNavigateToProfile,
                    )
                }

                // Refund Policy Link
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .clickable {
                            showEventInfoSheet = false
                            showRefundPolicy = true
                        }
                        .padding(vertical = 8.dp),
                ) {
                    Kicker(text = "Refund Policy", color = Slate)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(text = "→", color = Slate, style = Typography.labelMedium)
                }
            }
        }
    }

    if (showInbox) {
        RunnerInboxSheet(
            messages = inboxMessages,
            onDismiss = { showInbox = false },
            onMarkRead = { inboxViewModel.markRead(it) },
            onMarkAllRead = { inboxViewModel.markAllRead() },
            onRemove = { inboxViewModel.remove(it) },
            onOpenOrder = { orderId ->
                showInbox = false
                onOpenOrder(orderId)
            },
            fetchError = inboxViewModel.fetchError.collectAsState().value,
            onRetry = { inboxViewModel.fetchMessages() },
        )
    }
}

/**
 * Live-arrival strip above the grid. Port of the website's cockpit banner
 * (`event-cockpit.tsx`): a count of photos that landed while the runner was
 * looking, or a manual refresh once the socket has stopped healing itself.
 *
 * Tapping starts a fresh diversity-ranked snapshot and returns to the top, so
 * live uploads never jump ahead of an unrepresented photographer by themselves.
 */
@Composable
private fun LivePhotosBanner(
    newPhotoCount: Int,
    liveState: WsState,
    onJumpToTop: () -> Unit,
    onRetry: () -> Unit,
) {
    // Only nag about the connection once it has stopped quietly retrying;
    // a blip mid-race is not the runner's problem to solve.
    val giveUp = liveState is WsState.Failed && liveState.attempts > QpWebSocket.MAX_QUIET_ATTEMPTS
    AnimatedVisibility(
        visible = newPhotoCount > 0 || giveUp,
        enter = slideInVertically { -it } + fadeIn(),
        exit = slideOutVertically { -it } + fadeOut(),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 48.dp)
                .clip(PillShape)
                .clickable { if (giveUp) onRetry() else onJumpToTop() }
                .padding(vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            if (giveUp) {
                Icon(
                    imageVector = Icons.Default.Refresh,
                    contentDescription = null,
                    tint = Slate,
                    modifier = Modifier.size(14.dp),
                )
                Spacer(modifier = Modifier.width(6.dp))
                Kicker("Connection lost · Refresh", color = Slate)
            } else {
                Text(
                    text = "$newPhotoCount",
                    style = NumeralStyle.copy(fontSize = 12.sp),
                    color = Fresh,
                )
                Spacer(modifier = Modifier.width(6.dp))
                Kicker(
                    text = if (newPhotoCount == 1) "new photo · jump to top" else "new photos · jump to top",
                    color = Fresh,
                )
                Spacer(modifier = Modifier.width(4.dp))
                Icon(
                    imageVector = Icons.Default.KeyboardArrowUp,
                    contentDescription = null,
                    tint = Fresh,
                    modifier = Modifier.size(16.dp),
                )
            }
        }
    }
}

// Pre-race-day stand-in for the search cockpit. Faithful port of the website's
// UpcomingEventNotice (events/[slug]/page.tsx): 16:9 cover, Fresh "OPENS ·
// [date]" kicker, name, city, venue, and the race-day + four-day-window copy.
// The runner sees why there's nothing to search yet instead of an empty grid.
@Composable
private fun UpcomingEventNotice(
    event: EventDto,
    onBack: () -> Unit,
) {
    val dateLabel = remember(event.date) { formatUpcomingDate(event.date) }
    val cityUpper = remember(event.location) {
        event.location.substringAfterLast(',').trim().uppercase()
    }

    Column(modifier = Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) {
                Icon(
                    imageVector = Icons.Default.ArrowBack,
                    contentDescription = "Back to Events",
                    tint = Ink,
                )
            }
            Text(
                text = "ALL EVENTS",
                style = Typography.labelMedium,
                color = Slate,
            )
        }
        Spacer(modifier = Modifier.height(12.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(16f / 9f)
                .clip(QpCardShape)
                .background(Ink),
            contentAlignment = Alignment.Center,
        ) {
            if (!event.bannerUrl.isNullOrEmpty()) {
                AsyncImage(
                    model = RetrofitClient.resolveImageUrl(event.bannerUrl),
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop,
                )
            } else {
                Text(
                    text = event.name,
                    style = Typography.titleLarge,
                    color = Bone.copy(alpha = 0.25f),
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(horizontal = 24.dp),
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
        // The one Fresh element in this viewport — the notice has no CTA, so the
        // date kicker carries the accent, as it does on the web.
        Kicker(text = "Opens · $dateLabel", color = Fresh)
        Spacer(modifier = Modifier.height(12.dp))
        // displayMedium, not the Anton hero style: the event name is
        // user-generated text and uppercasing it in a condensed display face
        // reads wrong (flagged during the Finish Line migration).
        Text(
            text = event.name,
            style = Typography.displayMedium,
            color = Ink,
        )
        Spacer(modifier = Modifier.height(12.dp))
        Kicker(text = cityUpper, color = Slate)
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = event.location,
            style = Typography.bodyMedium,
            color = InkSoft,
        )
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            text = "The gallery and runner search open on race day. " +
                "Photographers have a four-day window from race day to upload — " +
                "check back then to find your photos.",
            style = Typography.bodyMedium,
            color = InkSoft,
        )
    }
}

// "Saturday, October 3, 2026" — matches the website's toLocaleDateString with
// weekday/month/day/year. Falls back to the raw ISO date if it can't parse.
private fun formatUpcomingDate(iso: String): String = try {
    LocalDate.parse(iso).format(
        DateTimeFormatter.ofPattern("EEEE, MMMM d, yyyy", Locale.US)
    )
} catch (e: Exception) {
    iso
}

// Runner opt-in card — "Get notified when your photos are ready". Mirrors the
// website's PhotoAlertToggle. GhostCta (not PrimaryCta) keeps the single Fresh
// accent for the page's real highlight; the registered state uses a
// SuccessGreen StatusChip, a distinct token from the Fresh CTA.
@Composable
private fun PhotoAlertCard(
    state: PhotoAlertUiState,
    onToggle: (Boolean) -> Unit,
    onAddSelfie: () -> Unit,
) {
    when (state) {
        is PhotoAlertUiState.Loading -> {
            LoadingSkeleton(
                shape = QpCardShape,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(96.dp),
            )
        }
        is PhotoAlertUiState.NeedsSelfie -> {
            QpCard(modifier = Modifier.fillMaxWidth()) {
                Kicker("Photo alerts", color = Slate)
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Get notified when your photos are ready",
                    style = Typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Ink,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = "Add a selfie and we'll email you the moment we spot you.",
                    style = Typography.bodyMedium,
                    color = SlateSoft,
                )
                Spacer(Modifier.height(16.dp))
                GhostCta(
                    text = "Add a selfie →",
                    onClick = onAddSelfie,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
        is PhotoAlertUiState.Ready -> {
            QpCard(modifier = Modifier.fillMaxWidth()) {
                Kicker("Photo alerts", color = Slate)
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Get notified when your photos are ready",
                    style = Typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Ink,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = if (state.registered)
                        "You're on the list — we'll email you when your photos land."
                    else
                        "We'll email you the moment your photos land.",
                    style = Typography.bodyMedium,
                    color = SlateSoft,
                )
                if (state.message != null) {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = state.message,
                        style = Typography.bodySmall,
                        color = ErrorRed,
                    )
                }
                Spacer(Modifier.height(16.dp))
                if (state.registered) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        StatusChip(text = "Notifications on", tone = StatusTone.Approved)
                        Spacer(Modifier.weight(1f))
                        Text(
                            text = "Turn off",
                            style = Typography.labelMedium,
                            color = Slate,
                            modifier = Modifier
                                .clip(PillShape)
                                .clickable(enabled = !state.updating) { onToggle(false) }
                                .padding(horizontal = 10.dp, vertical = 8.dp),
                        )
                    }
                } else {
                    GhostCta(
                        text = "Notify me when ready",
                        onClick = { onToggle(true) },
                        enabled = !state.updating,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
        }
    }
}

// Compact pill rendered on each photo tile (bottom-right) for inline cart
// actions. Mirrors website photo-mosaic-tile.tsx: Bone backdrop with Ink text
// in default state, Fresh fill with white text when `filled = true` (used for
// the ✓-cart confirmation state). Tile-relative — kept narrow so two pills
// fit comfortably even at 2-column mobile widths.
@Composable
private fun TileActionPill(
    label: String,
    filled: Boolean,
    onClick: () -> Unit,
) {
    val bg = if (filled) Fresh else Bone.copy(alpha = 0.92f)
    val fg = if (filled) Color.White else Ink
    Box(
        modifier = Modifier
            // ponytail: 44dp touch target (was ~24dp) — one step under the 48dp
            // guideline to keep the tile corner uncluttered; the lightbox CTAs
            // are the 48dp primary path. Label lifted off the 9sp floor too.
            .heightIn(min = 44.dp)
            .clip(PillShape)
            .background(bg)
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp),
        contentAlignment = Alignment.Center,
    ) {
        ArrowLabel(
            text = label.uppercase(),
            color = fg,
            style = Typography.labelSmall,
            fontSize = 11.sp,
            iconSize = 12.dp,
        )
    }
}

@Composable
private fun PhotoGridSkeleton() {
    Column(modifier = Modifier.fillMaxWidth()) {
        repeat(2) { rowIndex ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                repeat(2) {
                    LoadingSkeleton(
                        shape = QpCardShape,
                        modifier = Modifier
                            .weight(1f)
                            .aspectRatio(0.85f),
                    )
                }
            }
            if (rowIndex < 1) Spacer(Modifier.height(12.dp))
        }
    }
}

@Composable
private fun StatusChip(state: EventState, modifier: Modifier = Modifier) {
    val (label, accent) = when (state) {
        EventState.LIVE -> "UPLOADING" to Fresh
        EventState.UPCOMING -> "UPCOMING" to Bone
        EventState.OPEN -> "READY" to Bone
        EventState.PAST -> "ARCHIVE" to Bone.copy(alpha = 0.7f)
    }
    Surface(
        shape = RoundedCornerShape(percent = 100),
        color = Ink.copy(alpha = 0.75f),
        modifier = modifier
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Box(modifier = Modifier.size(6.dp).clip(CircleShape).background(accent))
            Text(
                text = label,
                style = Typography.labelSmall.copy(fontWeight = FontWeight.Bold, letterSpacing = 0.5.sp),
                color = Bone
            )
        }
    }
}
