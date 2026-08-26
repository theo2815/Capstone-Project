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
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material.icons.filled.List
import androidx.compose.material3.*
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.QpWebSocket
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.WsState
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.Locale
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.foundation.shape.CircleShape
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.quickpitik.mobile.data.local.SessionManager
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RunnerGalleryScreen(
    viewModel: RunnerGalleryViewModel,
    cartViewModel: CartViewModel,
    inboxViewModel: RunnerInboxViewModel,
    onNavigateToOrders: () -> Unit,
    onNavigateToProfile: () -> Unit,
    onNavigateToSettings: () -> Unit,
    onNavigateBack: () -> Unit,
    onOpenOrder: (String) -> Unit,
    onOpenPhotographer: (String) -> Unit,
    onLogout: () -> Unit
) {
    // rememberSaveable: the typed bib + chosen search tab survive rotation.
    var bibSearchQuery by rememberSaveable { mutableStateOf("") }
    var activeSearchTab by rememberSaveable { mutableStateOf(0) } // 0 = Selfie, 1 = Bib Number
    var selectedPhotoForDetail by remember { mutableStateOf<PhotoDto?>(null) }

    val inboxMessages by inboxViewModel.messages.collectAsState()
    val inboxUnread by inboxViewModel.unreadCount.collectAsState()
    var showInbox by remember { mutableStateOf(false) }
    var showRefundPolicy by remember { mutableStateOf(false) }

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

    // Load the "notify me when ready" opt-in state whenever a new event is
    // selected (any lifecycle state — upcoming events are the main use case).
    // RUNNER-gated endpoint — skipped in photographer runner-view.
    if (isTrueRunner) LaunchedEffect(activeEvent?.slug) {
        activeEvent?.slug?.let { viewModel.loadPhotoAlert(it) }
    }

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
                        Text(
                            text = "GALLERY HUB",
                            style = Typography.labelMedium,
                            color = Slate
                        )
                        Text(
                            text = "QuickPitik",
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Ink
                        )
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        if (isTrueRunner) {
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
                                DropdownMenuItem(
                                    text = { Text("Sign Out", color = ErrorRed) },
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
                item(span = { GridItemSpan(maxLineSpan) }) {
                    Card(
                        shape = QpCardShape,
                        colors = CardDefaults.cardColors(containerColor = BoneDeep),
                        border = BorderStroke(1.dp, Line),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.fillMaxWidth()) {
                            if (!activeEvent?.bannerUrl.isNullOrEmpty()) {
                                AsyncImage(
                                    model = RetrofitClient.resolveImageUrl(activeEvent?.bannerUrl),
                                    contentDescription = "Event Banner",
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(160.dp)
                                        .clip(RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp)),
                                    contentScale = ContentScale.Crop
                                )
                            }

                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                IconButton(
                                    onClick = {
                                        viewModel.clearSelectedEvent()
                                        onNavigateBack()
                                    },
                                    modifier = Modifier.padding(end = 8.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.ArrowBack,
                                        contentDescription = "Back to Events",
                                        tint = Ink
                                    )
                                }
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = "SELECTED MARATHON",
                                        style = Typography.labelSmall,
                                        color = Slate
                                    )
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = activeEvent?.name ?: "",
                                        style = Typography.titleMedium,
                                        fontWeight = FontWeight.Bold,
                                        color = Ink
                                    )
                                    Text(
                                        text = activeEvent?.location ?: "",
                                        style = Typography.bodySmall,
                                        color = SlateSoft
                                    )
                                }
                            }
                        }
                    }
                }
                // Pre-event opt-in — mirrors the website cockpit's PhotoAlertToggle.
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
                // Pre-purchase refund disclosure — port of the web cockpit's
                // "Refund Policy →" kicker (event-cockpit.tsx). Read-only.
                item(span = { GridItemSpan(maxLineSpan) }) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .clickable { showRefundPolicy = true }
                            .padding(vertical = 8.dp),
                    ) {
                        Kicker(text = "Refund Policy", color = Slate)
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(text = "→", color = Slate, style = Typography.labelMedium)
                    }
                }

                // AI Search Selector Cards (Selfie vs Bib)
                item(span = { GridItemSpan(maxLineSpan) }) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // Selfie Match Card
                        Card(
                            onClick = { activeSearchTab = 0 },
                            border = BorderStroke(
                                width = 1.5.dp,
                                color = if (activeSearchTab == 0) Ink else Line
                            ),
                            colors = CardDefaults.cardColors(
                                containerColor = if (activeSearchTab == 0) BoneDeep else Bone
                            ),
                            shape = QpCardShape,
                            modifier = Modifier.weight(1f)
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Icon(Icons.Default.Face, contentDescription = "Selfie", tint = Fresh)
                                Spacer(modifier = Modifier.height(12.dp))
                                Text("Selfie Match", style = Typography.titleMedium, color = Ink)
                                Text("AI Face Search", style = Typography.bodyMedium, color = SlateSoft)
                            }
                        }

                        // Bib Number Search Card
                        Card(
                            onClick = { activeSearchTab = 1 },
                            border = BorderStroke(
                                width = 1.5.dp,
                                color = if (activeSearchTab == 1) Ink else Line
                            ),
                            colors = CardDefaults.cardColors(
                                containerColor = if (activeSearchTab == 1) BoneDeep else Bone
                            ),
                            shape = QpCardShape,
                            modifier = Modifier.weight(1f)
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Icon(Icons.Default.Search, contentDescription = "Bib", tint = Fresh)
                                Spacer(modifier = Modifier.height(12.dp))
                                Text("Bib Lookup", style = Typography.titleMedium, color = Ink)
                                Text("Search by Number", style = Typography.bodyMedium, color = SlateSoft)
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
                                modifier = Modifier.padding(24.dp),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Text(
                                    text = "Find photos of yourself instantly with face recognition.",
                                    textAlign = TextAlign.Center,
                                    style = Typography.bodyMedium,
                                    color = InkSoft
                                )
                                Spacer(modifier = Modifier.height(16.dp))
                                // Primary: search with a saved library selfie
                                PrimaryCta(
                                    text = "Search with stored selfie",
                                    onClick = { viewModel.searchByStoredSelfie() },
                                    modifier = Modifier.fillMaxWidth(),
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                Row(
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    // Take a selfie now — opens the camera via a MediaStore URI
                                    GhostCta(
                                        text = "Take a selfie",
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
                                                // Swallow — camera unavailable; user can still use Upload/Stored.
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
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }

                // Watermarked Photo Stream Title with Reset Button
                item(span = { GridItemSpan(maxLineSpan) }) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.Top
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Kicker("Matched photos")
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Watermarked previews — buy to unlock the full-resolution shot.",
                                style = Typography.bodySmall,
                                color = SlateSoft,
                            )
                        }
                        if (isFiltered || searchState is PhotosSearchState.Error) {
                            Text(
                                text = "Reset",
                                style = Typography.labelMedium,
                                color = Fresh,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier
                                    .padding(start = 12.dp, top = 2.dp)
                                    .clickable {
                                        bibSearchQuery = ""
                                        viewModel.clearFilter()
                                    }
                            )
                        } else {
                            // Back to My Photos — one tap re-runs the stored-selfie
                            // face match (mirrors the website's "My photos" control).
                            Text(
                                text = "My photos",
                                style = Typography.labelMedium,
                                color = Fresh,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier
                                    .padding(start = 12.dp, top = 2.dp)
                                    .clickable {
                                        bibSearchQuery = ""
                                        viewModel.searchByStoredSelfie()
                                    }
                            )
                        }
                    }
                }

                item(span = { GridItemSpan(maxLineSpan) }) {
                    LivePhotosBanner(
                        newPhotoCount = newPhotoCount,
                        liveState = liveState,
                        onJumpToTop = {
                            viewModel.resetNewPhotoCount()
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
                            ErrorView(
                                message = state.message,
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                    }
                    is PhotosSearchState.Success -> {
                        if (state.photos.isEmpty()) {
                            item(span = { GridItemSpan(maxLineSpan) }) {
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(vertical = 32.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Text(
                                        text = "No matched photos found for this event.\nTry another search parameter or selfie scan!",
                                        color = SlateSoft,
                                        textAlign = TextAlign.Center,
                                        style = Typography.bodyMedium
                                    )
                                }
                            }
                        } else {
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
                                        Box(
                                            modifier = Modifier
                                                .fillMaxSize()
                                                .background(Color.Black.copy(alpha = 0.3f))
                                        )
                                    }

                                    // Premium transparent watermark overlay (Fulfills NFR-P-2 security preview)
                                    Box(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .background(Color.Black.copy(alpha = 0.04f)),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Text(
                                            text = "QUICKPITIK\nPREVIEW",
                                            color = Color.White.copy(alpha = 0.35f),
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold,
                                            textAlign = TextAlign.Center,
                                            letterSpacing = 1.5.sp
                                        )
                                    }

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

    if (selectedPhotoForDetail != null) {
        val photo = selectedPhotoForDetail!!
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
        )
    }
}

/**
 * Live-arrival strip above the grid. Port of the website's cockpit banner
 * (`event-cockpit.tsx`): a count of photos that landed while the runner was
 * looking, or a manual refresh once the socket has stopped healing itself.
 *
 * The grid refreshes on its own — this is purely "something changed below the
 * fold", so it stays a quiet mono line rather than a filled CTA.
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
        Text(
            text = event.name,
            style = Typography.displayLarge,
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
            .clip(PillShape)
            .background(bg)
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 6.dp),
    ) {
        ArrowLabel(
            text = label.uppercase(),
            color = fg,
            style = Typography.labelSmall,
            fontSize = 9.sp,
            iconSize = 11.dp,
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
