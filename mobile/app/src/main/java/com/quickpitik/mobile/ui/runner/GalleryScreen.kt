package com.quickpitik.mobile.ui.runner

import android.content.ContentValues
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.MutableTransitionState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
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
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.BottomSheetDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
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
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.QpWebSocket
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.WsState
import com.quickpitik.mobile.ui.theme.ArrowLabel
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorView
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.HeroText
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.LineStrong
import com.quickpitik.mobile.ui.theme.LoadingSkeleton
import com.quickpitik.mobile.ui.theme.MosaicTileShape
import com.quickpitik.mobile.ui.theme.NumeralStyle
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.QpHaptic
import com.quickpitik.mobile.ui.theme.SecureScreen
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.SurfaceWhite
import com.quickpitik.mobile.ui.theme.Typography
import com.quickpitik.mobile.ui.theme.rememberQpHaptic
import kotlinx.coroutines.launch

// The runner Event page — a phone-sized port of the website's events/[slug]
// page. Two modes, one screen: the search COCKPIT (card over a dimmed photo
// wall, About strip below) and the BROWSE wall (hero header, sticky search
// pill, 2-up tiles, load-more). Cockpit-mode pieces live in EventCockpit.kt.
enum class GalleryMode { Cockpit, Browse }

// The API's default page size for /events/{slug}/photos — the "Load N more"
// label is derived from it (loadMorePhotos never passes an explicit limit).
private const val PHOTO_PAGE_SIZE = 100L

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun RunnerGalleryScreen(
    viewModel: RunnerGalleryViewModel,
    cartViewModel: CartViewModel,
    inboxViewModel: RunnerInboxViewModel,
    savedEventsViewModel: SavedEventsViewModel,
    onNavigateBack: () -> Unit,
    onOpenOrder: (String) -> Unit,
    onOpenPhotographer: (String) -> Unit,
    onLogout: () -> Unit
) {
    // Runner browse surface — every photo here is an unpurchased preview, so
    // screenshots/recording/casting are blocked for the whole screen.
    SecureScreen()

    // rememberSaveable: mode, panel and the typed bib survive rotation.
    var mode by rememberSaveable { mutableStateOf(GalleryMode.Cockpit) }
    var panelMode by rememberSaveable { mutableStateOf(SearchPanelMode.Bib) }
    var bibInput by rememberSaveable { mutableStateOf("") }
    // The bib the last search ran with — drives the browse header + empty copy.
    var submittedBib by rememberSaveable { mutableStateOf("") }
    var showFindSheet by rememberSaveable { mutableStateOf(false) }
    // Photo Alerts "Add a selfie →": the chooser sheet, and whether the next
    // picked image is library-only (no search). Saveable so the camera
    // activity round trip can't forget which intent launched it.
    var showAddSelfieSheet by rememberSaveable { mutableStateOf(false) }
    var selfieAddOnly by rememberSaveable { mutableStateOf(false) }
    // A selfie search is in flight from the panel; the screen enters Browse
    // only once it succeeds, so an error never strands the runner on an empty
    // wall (the panel shows the message instead).
    var faceSearchPending by remember { mutableStateOf(false) }
    var faceSearchError by remember { mutableStateOf<String?>(null) }
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
    val eventDetail by viewModel.eventDetail.collectAsState()
    val librarySelfies by viewModel.selfies.collectAsState()
    val saveToLibrary by viewModel.saveSelfieToLibrary.collectAsState()
    val saveNotice by viewModel.selfieSaveNotice.collectAsState()
    // Hoisted so the live-photos pill can jump the runner back to the top when
    // new shots land while they're scrolled down the wall.
    val listState = rememberLazyListState()
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
    // Hoisted so the tile pills can read in-cart state for the ✓ Cart flip
    // without each tile collecting its own copy.
    val cartItems by cartViewModel.cartItems.collectAsState()

    // Live selfie capture (camera) + gallery pick — a MediaStore URI handed to
    // TakePicture(), then face-search the bytes.
    val context = LocalContext.current
    var pendingSelfieUri by remember { mutableStateOf<Uri?>(null) }
    fun startFaceSearch(run: () -> Unit) {
        faceSearchError = null
        faceSearchPending = true
        run()
    }
    // One landing for both launchers: the alert card's sheet saves to the
    // library and stops; the search panel saves-and-searches.
    fun onSelfiePicked(uri: Uri) {
        if (selfieAddOnly) viewModel.addSelfieToLibrary(uri)
        else startFaceSearch { viewModel.searchBySelfieUri(uri) }
    }
    val selfieCameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) pendingSelfieUri?.let { onSelfiePicked(it) }
    }
    val selfieGalleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { onSelfiePicked(it) }
    }
    val launchCamera = {
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
    }

    // One selfie-library request feeds both the picker and the runner-only
    // photo-alert check.
    LaunchedEffect(activeEvent?.slug, isTrueRunner) {
        viewModel.loadGalleryMetadata(activeEvent?.slug.takeIf { isTrueRunner })
    }

    // Face search resolution: Success → the wall; Error → stays in the panel.
    LaunchedEffect(searchState) {
        if (!faceSearchPending) return@LaunchedEffect
        when (val s = searchState) {
            is PhotosSearchState.Success -> {
                faceSearchPending = false
                submittedBib = ""
                showFindSheet = false
                mode = GalleryMode.Browse
            }
            is PhotosSearchState.Error -> {
                faceSearchPending = false
                faceSearchError = s.message
            }
            else -> Unit
        }
    }

    // System back from the wall returns to the cockpit (website: "← Back").
    BackHandler(enabled = mode == GalleryMode.Browse) { mode = GalleryMode.Cockpit }

    val submitBib = {
        val clean = bibInput.trim().uppercase()
        if (clean.isNotEmpty()) {
            bibInput = clean
            submittedBib = clean
            showFindSheet = false
            viewModel.searchByBib(clean)
            mode = GalleryMode.Browse
        }
    }
    val clearFilter = {
        bibInput = ""
        submittedBib = ""
        faceSearchError = null
        viewModel.clearFilter()
    }
    val browseAll = {
        if (isFiltered) clearFilter()
        mode = GalleryMode.Browse
    }
    val backToEvents = {
        viewModel.clearSelectedEvent()
        onNavigateBack()
    }

    val selfiePanelState = SelfiePanelState(
        selfies = librarySelfies,
        saveToLibrary = saveToLibrary,
        canSave = isTrueRunner && librarySelfies.size < SELFIE_MAX,
        saveNotice = saveNotice,
        matching = faceSearchPending,
        error = faceSearchError,
    )
    val panelCallbacks = SearchPanelCallbacks(
        onBibChange = { bibInput = it },
        onSubmitBib = submitBib,
        onSwitchToSelfie = { panelMode = SearchPanelMode.Selfie },
        onSwitchToBib = { faceSearchError = null; panelMode = SearchPanelMode.Bib },
        onTakeSelfie = { selfieAddOnly = false; launchCamera() },
        onUploadSelfie = { selfieAddOnly = false; selfieGalleryLauncher.launch("image/*") },
        onMatchAllSelfies = { startFaceSearch { viewModel.searchWithAllSelfies() } },
        onPickSelfie = { s -> startFaceSearch { viewModel.searchBySelfieId(s.id) } },
        onSaveToLibraryChange = { viewModel.setSaveSelfieToLibrary(it) },
    )
    val photoAlertCard: @Composable () -> Unit = {
        PhotoAlertCard(
            state = photoAlert,
            onToggle = { register ->
                activeEvent?.slug?.let { viewModel.togglePhotoAlert(it, register) }
            },
            // In place — never a Profile detour (2026-09-02 ADR: that detour
            // was the hassle). Covers all three card placements.
            onAddSelfie = { showAddSelfieSheet = true },
        )
    }

    Surface(modifier = Modifier.fillMaxSize(), color = Bone) {
        val event = activeEvent
        Box(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding(),
        ) {
            when {
                event == null -> {
                    LoadingSkeleton(
                        shape = QpCardShape,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(24.dp)
                            .height(220.dp),
                    )
                }
                deriveEventState(event.date) == EventState.UPCOMING -> {
                    // Pre-race-day: no gallery, no search. Port of the website's
                    // UpcomingEventNotice branch in events/[slug]/page.tsx.
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .verticalScroll(rememberScrollState())
                            .padding(24.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp),
                    ) {
                        UpcomingEventNotice(event = event, onBack = backToEvents)
                        // Pre-event opt-in — most useful before race day, when
                        // there's nothing to search yet. RUNNER-gated.
                        if (isTrueRunner) photoAlertCard()
                    }
                }
                else -> AnimatedContent(
                    targetState = mode,
                    transitionSpec = {
                        (fadeIn(tween(220)) + slideInVertically(tween(220)) { it / 14 })
                            .togetherWith(fadeOut(tween(140)))
                    },
                    label = "gallery-mode",
                ) { target ->
                    when (target) {
                        GalleryMode.Cockpit -> CockpitScreen(
                            backdropPhotos = (searchState as? PhotosSearchState.Success)?.photos.orEmpty(),
                            onBack = backToEvents,
                            topActions = {
                                if (isTrueRunner) {
                                    val savedIds by savedEventsViewModel.savedIds.collectAsState()
                                    val saved = event.id in savedIds
                                    IconButton(onClick = { savedEventsViewModel.toggle(event) }) {
                                        Icon(
                                            imageVector = if (saved) Icons.Default.Favorite
                                            else Icons.Default.FavoriteBorder,
                                            contentDescription = if (saved) "Remove from saved" else "Save event",
                                            tint = if (saved) Fresh else Slate,
                                        )
                                    }
                                    RunnerInboxBell(
                                        messageCount = inboxMessages.size,
                                        unreadCount = inboxUnread,
                                        onClick = { showInbox = true },
                                    )
                                }
                            },
                            card = {
                                if (event.photoCount == 0) {
                                    EmptyCockpitCard(eventName = event.name)
                                } else {
                                    CockpitCard(
                                        eventName = event.name,
                                        heroLine1 = "Find your",
                                        heroLine2 = "photos.",
                                    ) {
                                        SearchPanel(
                                            mode = panelMode,
                                            bib = bibInput,
                                            photoCount = event.photoCount,
                                            selfie = selfiePanelState,
                                            callbacks = panelCallbacks,
                                        )
                                    }
                                }
                            },
                            alertCard = if (isTrueRunner) photoAlertCard else null,
                            browseLabel = if (event.photoCount == 0) "Browse all photos"
                            else "Browse all ${"%,d".format(event.photoCount)} photos",
                            onBrowseAll = browseAll,
                            aboutStrip = {
                                AboutStrip(
                                    event = event,
                                    detail = eventDetail,
                                    onRefundPolicy = { showRefundPolicy = true },
                                )
                            },
                        )
                        GalleryMode.Browse -> BrowseScreen(
                            eventName = event.name,
                            eventState = deriveEventState(event.date),
                            state = searchState,
                            isFiltered = isFiltered,
                            submittedBib = submittedBib,
                            isTrueRunner = isTrueRunner,
                            hasLibrary = librarySelfies.isNotEmpty(),
                            listState = listState,
                            newPhotoCount = newPhotoCount,
                            liveState = liveState,
                            cartItemIds = cartItems.map { it.photoId }.toSet(),
                            onBack = { mode = GalleryMode.Cockpit },
                            onOpenSearch = { showFindSheet = true },
                            onClearFilter = clearFilter,
                            onMyPhotos = { startFaceSearch { viewModel.searchByStoredSelfie() } },
                            onRefundPolicy = { showRefundPolicy = true },
                            onJumpToTop = {
                                viewModel.refreshLivePhotos()
                                scope.launch { listState.animateScrollToItem(0) }
                            },
                            onRetryLive = { viewModel.retryLivePhotos() },
                            onRetry = { faceSearchError = null; viewModel.searchByBib(submittedBib) },
                            onLoadMore = { viewModel.loadMorePhotos() },
                            onOpenPhoto = { selectedPhotoForDetail = it },
                            onToggleCart = { photo, inCart ->
                                if (inCart) cartViewModel.removeFromCart(photo.id)
                                else cartViewModel.addToCart(photo, event.id, event.slug, event.name)
                            },
                            onBuyNow = { photo, inCart ->
                                if (inCart) {
                                    cartViewModel.openCheckoutSheet()
                                } else {
                                    cartViewModel.triggerExpressCheckout()
                                    cartViewModel.addToCart(photo, event.id, event.slug, event.name)
                                }
                            },
                            onBuyAll = { photos ->
                                photos.forEach { p ->
                                    if (cartItems.none { it.photoId == p.id }) {
                                        cartViewModel.addToCart(p, event.id, event.slug, event.name)
                                    }
                                }
                            },
                            alertCard = if (isTrueRunner) photoAlertCard else null,
                        )
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

    // Browse-mode search — the website's FindPhotosModal, as a sheet. Same
    // panel as the cockpit card, so the two can never drift.
    if (showFindSheet && activeEvent != null) {
        ModalBottomSheet(
            onDismissRequest = { showFindSheet = false },
            containerColor = Bone,
            dragHandle = { BottomSheetDefaults.DragHandle(color = Line) },
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .imePadding()
                    .padding(horizontal = 24.dp)
                    .padding(bottom = 32.dp),
            ) {
                Kicker(activeEvent?.name.orEmpty())
                Spacer(Modifier.height(16.dp))
                // Archivo here, not Anton — the website's modal headline is
                // font-display; the Anton hero belongs to the page itself.
                Text("Find your", style = Typography.displayMedium, color = Ink)
                Text("photos.", style = Typography.displayMedium, color = Fresh)
                SearchPanel(
                    mode = panelMode,
                    bib = bibInput,
                    photoCount = activeEvent?.photoCount ?: 0,
                    selfie = selfiePanelState,
                    callbacks = panelCallbacks,
                )
            }
        }
    }

    // Photo Alerts "Add a selfie →" chooser — library-only, no search, so it
    // works from the upcoming notice and the empty wall too.
    // ponytail: mirrors ProfileScreen's add sheet; extract when a third caller appears.
    if (showAddSelfieSheet) {
        ModalBottomSheet(
            onDismissRequest = { showAddSelfieSheet = false },
            containerColor = Bone,
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp)
                    .padding(bottom = 28.dp),
            ) {
                Kicker("Add a selfie")
                Spacer(Modifier.height(6.dp))
                Text(
                    text = "Face the camera, good light, no sunglasses or cap. " +
                        "${librarySelfies.size} of $SELFIE_MAX used.",
                    style = Typography.bodyMedium,
                    color = Slate,
                )
                Spacer(Modifier.height(16.dp))
                PrimaryCta(
                    text = "Take a selfie",
                    onClick = { showAddSelfieSheet = false; selfieAddOnly = true; launchCamera() },
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(8.dp))
                GhostCta(
                    text = "Upload a photo",
                    onClick = {
                        showAddSelfieSheet = false
                        selfieAddOnly = true
                        selfieGalleryLauncher.launch("image/*")
                    },
                    modifier = Modifier.fillMaxWidth(),
                )
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

/* ─────────────── COCKPIT MODE ─────────────── */

@Composable
private fun CockpitScreen(
    backdropPhotos: List<PhotoDto>,
    onBack: () -> Unit,
    topActions: @Composable androidx.compose.foundation.layout.RowScope.() -> Unit,
    card: @Composable () -> Unit,
    alertCard: (@Composable () -> Unit)?,
    browseLabel: String,
    onBrowseAll: () -> Unit,
    aboutStrip: @Composable () -> Unit,
) {
    // Website: the cockpit section is min-h-[78vh] so the card sits mid-screen
    // and the About strip waits below the fold.
    val minSectionHeight = (LocalConfiguration.current.screenHeightDp * 0.78f).dp
    // Card entrance — the website's fade-up. Runs once per composition.
    val entrance = remember { MutableTransitionState(false).apply { targetState = true } }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
    ) {
        EventCockpitTopRow(onBack = onBack, trailing = topActions)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = minSectionHeight),
            contentAlignment = Alignment.Center,
        ) {
            DimWall(photos = backdropPhotos, modifier = Modifier.matchParentSize())
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 40.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                AnimatedVisibility(
                    visibleState = entrance,
                    enter = fadeIn(tween(280)) + slideInVertically(tween(280)) { it / 10 },
                    exit = fadeOut(),
                ) {
                    card()
                }
                if (alertCard != null) {
                    Spacer(Modifier.height(16.dp))
                    alertCard()
                }
                Spacer(Modifier.height(24.dp))
                BrowseAllLink(label = browseLabel, onClick = onBrowseAll)
            }
        }
        aboutStrip()
        // Room for the floating nav pill.
        Spacer(Modifier.height(24.dp))
    }
}

/* ─────────────── BROWSE MODE ─────────────── */

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun BrowseScreen(
    eventName: String,
    eventState: EventState,
    state: PhotosSearchState,
    isFiltered: Boolean,
    submittedBib: String,
    isTrueRunner: Boolean,
    hasLibrary: Boolean,
    listState: androidx.compose.foundation.lazy.LazyListState,
    newPhotoCount: Int,
    liveState: WsState,
    cartItemIds: Set<String>,
    onBack: () -> Unit,
    onOpenSearch: () -> Unit,
    onClearFilter: () -> Unit,
    onMyPhotos: () -> Unit,
    onRefundPolicy: () -> Unit,
    onJumpToTop: () -> Unit,
    onRetryLive: () -> Unit,
    onRetry: () -> Unit,
    onLoadMore: () -> Unit,
    onOpenPhoto: (PhotoDto) -> Unit,
    onToggleCart: (PhotoDto, Boolean) -> Unit,
    onBuyNow: (PhotoDto, Boolean) -> Unit,
    onBuyAll: (List<PhotoDto>) -> Unit,
    alertCard: (@Composable () -> Unit)?,
) {
    val isBibFilter = isFiltered && submittedBib.isNotEmpty()
    val isFaceFilter = isFiltered && submittedBib.isEmpty()
    val success = state as? PhotosSearchState.Success
    val photos = success?.photos.orEmpty()
    val total = success?.total ?: 0L
    val haptic = rememberQpHaptic()

    LazyColumn(
        state = listState,
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(bottom = 24.dp),
    ) {
        item(key = "header") {
            BrowseHeader(
                eventName = eventName,
                state = state,
                isBibFilter = isBibFilter,
                isFaceFilter = isFaceFilter,
                submittedBib = submittedBib,
                photoCount = photos.size,
                total = total,
                onBack = onBack,
                onRefundPolicy = onRefundPolicy,
            )
        }
        stickyHeader(key = "search-bar") {
            Column(modifier = Modifier.background(Bone)) {
                FindPhotosBar(
                    label = when {
                        isBibFilter -> "Search · $submittedBib"
                        isFaceFilter -> "Search · selfie"
                        else -> "Find your photos"
                    },
                    filtered = isFiltered,
                    photoCount = total,
                    hasLibrary = hasLibrary && isTrueRunner,
                    onOpenSearch = onOpenSearch,
                    onClearFilter = onClearFilter,
                    onMyPhotos = onMyPhotos,
                )
                LivePhotosBanner(
                    newPhotoCount = newPhotoCount,
                    liveState = liveState,
                    onJumpToTop = onJumpToTop,
                    onRetry = onRetryLive,
                )
                HorizontalDivider(color = Line)
            }
        }
        when (state) {
            is PhotosSearchState.Loading -> item(key = "skeleton") { PhotoGridSkeleton() }
            is PhotosSearchState.Error -> item(key = "error") {
                // Retry re-runs the submitted bib query (or the plain browse
                // when empty) — the same recovery the website offers.
                ErrorView(
                    message = state.message,
                    onRetry = onRetry,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            is PhotosSearchState.Success -> {
                if (photos.isEmpty()) {
                    item(key = "empty") {
                        when {
                            isBibFilter -> BibEmptyResult(
                                bib = submittedBib,
                                eventName = eventName,
                                eventState = eventState,
                                onClear = onClearFilter,
                            )
                            isFaceFilter -> FaceEmptyResult(onClear = onClearFilter)
                            else -> GalleryEmptyResult(alertCard = alertCard)
                        }
                    }
                } else {
                    // "Buy all N · ₱X →" — web BuyAllBar parity: one tap adds
                    // every visible match. Only on a filtered set (your
                    // matches, not the whole wall) and only for true runners
                    // (cart is RUNNER-gated server-side). Inline under the
                    // sticky bar rather than pinned to the bottom, where the
                    // floating cart pill already lives.
                    if (isFiltered && isTrueRunner) {
                        item(key = "buy-all") {
                            val allInCart = photos.all { it.id in cartItemIds }
                            val bulkTotal = photos.sumOf { it.price }
                            PrimaryCta(
                                text = when {
                                    allInCart -> "Added · ${photos.size} in cart ✓"
                                    photos.size == 1 -> "Buy 1 · ₱${"%,.0f".format(bulkTotal)} →"
                                    else -> "Buy all ${photos.size} · ₱${"%,.0f".format(bulkTotal)} →"
                                },
                                enabled = !allInCart,
                                onClick = { onBuyAll(photos) },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(start = 24.dp, end = 24.dp, top = 16.dp, bottom = 4.dp),
                            )
                        }
                    }
                    val rows = photos.chunked(2)
                    items(rows.size, key = { rows[it].first().id }) { index ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 24.dp, vertical = 6.dp)
                                .padding(top = if (index == 0) 6.dp else 0.dp),
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                        ) {
                            rows[index].forEach { photo ->
                                val inCart = photo.id in cartItemIds
                                PhotoTile(
                                    photo = photo,
                                    inCart = inCart,
                                    showCommerce = isTrueRunner,
                                    onOpen = { onOpenPhoto(photo) },
                                    onToggleCart = {
                                        if (!inCart) haptic(QpHaptic.CONFIRM)
                                        onToggleCart(photo, inCart)
                                    },
                                    onBuyNow = { onBuyNow(photo, inCart) },
                                    modifier = Modifier.weight(1f),
                                )
                            }
                            if (rows[index].size == 1) Spacer(Modifier.weight(1f))
                        }
                    }
                    item(key = "load-more") {
                        LoadMoreFooter(
                            shown = photos.size,
                            total = total,
                            loading = state.loadingMore,
                            suffix = when {
                                isBibFilter -> " · Bib $submittedBib"
                                isFaceFilter -> " · Selfie match"
                                else -> ""
                            },
                            onLoadMore = onLoadMore,
                        )
                    }
                }
            }
            PhotosSearchState.Idle -> Unit
        }
    }
}

@Composable
private fun BrowseHeader(
    eventName: String,
    state: PhotosSearchState,
    isBibFilter: Boolean,
    isFaceFilter: Boolean,
    submittedBib: String,
    photoCount: Int,
    total: Long,
    onBack: () -> Unit,
    onRefundPolicy: () -> Unit,
) {
    val anyFilter = isBibFilter || isFaceFilter
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 12.dp, end = 24.dp, top = 4.dp, bottom = 16.dp),
    ) {
        BackKicker(text = "Back", onClick = onBack)
        Spacer(Modifier.height(12.dp))
        Kicker(
            text = when {
                isBibFilter -> "$eventName · Bib $submittedBib"
                isFaceFilter -> "$eventName · Selfie match"
                else -> "$eventName · Gallery"
            },
            modifier = Modifier.padding(start = 12.dp),
        )
        Spacer(Modifier.height(12.dp))
        val hero = buildAnnotatedString {
            when {
                anyFilter && state is PhotosSearchState.Loading -> append("Searching…")
                anyFilter && photoCount == 0 -> append("No matches yet.")
                anyFilter -> {
                    append("We found ")
                    withStyle(SpanStyle(color = Fresh)) { append("%,d".format(total)) }
                    append(
                        if (isFaceFilter) (if (total == 1L) " match." else " matches.")
                        else (if (total == 1L) " photo." else " photos.")
                    )
                }
                else -> append("Browse ${"%,d".format(total)} photos.")
            }
        }
        HeroText(hero, modifier = Modifier.padding(start = 12.dp))
        Spacer(Modifier.height(12.dp))
        Text(
            text = when {
                isFaceFilter -> "These are the photos that match your selfie. Tap any to add to cart."
                isBibFilter -> "These are the photos matching your bib. Tap any to add to cart."
                else -> "Skim the wall, or open search anytime to find your bib."
            },
            style = Typography.bodyLarge,
            color = InkSoft,
            modifier = Modifier.padding(start = 12.dp),
        )
        Row(
            modifier = Modifier
                .heightIn(min = 48.dp)
                .clip(PillShape)
                .clickable(onClick = onRefundPolicy)
                .padding(horizontal = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            ArrowLabel("Refund policy →", color = SlateSoft)
        }
    }
}

/** The sticky search pill + its trailing action (website BrowseMode sticky bar). */
@Composable
private fun FindPhotosBar(
    label: String,
    filtered: Boolean,
    photoCount: Long,
    hasLibrary: Boolean,
    onOpenSearch: () -> Unit,
    onClearFilter: () -> Unit,
    onMyPhotos: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Surface(
            shape = PillShape,
            color = SurfaceWhite,
            border = BorderStroke(1.dp, LineStrong),
            modifier = Modifier
                .weight(1f)
                .heightIn(min = 48.dp)
                .clip(PillShape)
                .clickable(onClick = onOpenSearch),
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .padding(horizontal = 16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    imageVector = Icons.Default.Search,
                    contentDescription = null,
                    tint = Slate,
                    modifier = Modifier.size(18.dp),
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    text = label,
                    style = Typography.bodyMedium,
                    fontWeight = if (filtered) FontWeight.Bold else FontWeight.Medium,
                    color = if (filtered) Ink else Slate,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
        when {
            filtered -> GhostCta(text = "Clear", onClick = onClearFilter)
            hasLibrary -> GhostCta(text = "My photos", onClick = onMyPhotos)
            else -> Kicker("${"%,d".format(photoCount)} photos", color = SlateSoft)
        }
    }
}

/** One wall tile: watermarked preview + the `+ Cart` / `Buy →` pills (website PhotoMosaicTile). */
@Composable
private fun PhotoTile(
    photo: PhotoDto,
    inCart: Boolean,
    showCommerce: Boolean,
    onOpen: () -> Unit,
    onToggleCart: () -> Unit,
    onBuyNow: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .aspectRatio(0.85f)
            .clip(MosaicTileShape)
            .background(BoneDeep)
            .clickable(onClick = onOpen),
        contentAlignment = Alignment.Center,
    ) {
        if (photo.imageUrl != null) {
            AsyncImage(
                model = RetrofitClient.resolveImageUrl(photo.imageUrl),
                contentDescription = photo.bib?.let { "Race photo of bib $it" } ?: "Race photo",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop,
            )
        }
        // No client-side scrim or "PREVIEW" text: the backend bakes the
        // QuickPitik credit + photographer logo into imageUrl.
        // Cart is RUNNER-gated server-side — pills hidden in photographer runner-view.
        if (showCommerce) {
            Row(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(8.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                // Fresh marks only the in-cart state so a full wall doesn't
                // flood the accent; Buy is the heavier ink pill.
                TileActionPill(
                    label = if (inCart) "✓ Cart" else "+ Cart",
                    background = if (inCart) Fresh else SurfaceWhite.copy(alpha = 0.95f),
                    foreground = if (inCart) SurfaceWhite else Ink,
                    border = if (inCart) Color.Transparent else LineStrong,
                    onClick = onToggleCart,
                )
                TileActionPill(
                    label = "Buy →",
                    background = Ink.copy(alpha = 0.85f),
                    foreground = SurfaceWhite,
                    border = Color.Transparent,
                    onClick = onBuyNow,
                )
            }
        }
    }
}

// Compact pill rendered on each photo tile (bottom-right) for inline cart
// actions. Archivo bold, sentence case, like the website's tile pills.
@Composable
private fun TileActionPill(
    label: String,
    background: Color,
    foreground: Color,
    border: Color,
    onClick: () -> Unit,
) {
    Box(
        modifier = Modifier
            // ponytail: 44dp touch target — one step under the 48dp guideline
            // to keep the tile corner uncluttered; the lightbox CTAs are the
            // 48dp primary path.
            .heightIn(min = 44.dp)
            .clip(PillShape)
            .background(background)
            .border(1.dp, border, PillShape)
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp),
        contentAlignment = Alignment.Center,
    ) {
        ArrowLabel(
            text = label,
            color = foreground,
            style = Typography.titleSmall,
            fontWeight = FontWeight.Bold,
            fontSize = 12.sp,
            iconSize = 12.dp,
        )
    }
}

/** "Showing S of T" + a bone-deep "Load N more" pill (website LoadMoreButton). */
@Composable
private fun LoadMoreFooter(
    shown: Int,
    total: Long,
    loading: Boolean,
    suffix: String,
    onLoadMore: () -> Unit,
) {
    val remaining = (total - shown).coerceAtLeast(0L)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        if (remaining == 0L) {
            Kicker("All ${"%,d".format(total)} loaded", color = SlateSoft)
        } else {
            Kicker("Showing ${"%,d".format(shown)} of ${"%,d".format(total)}$suffix")
            Spacer(Modifier.height(12.dp))
            Surface(
                shape = PillShape,
                color = BoneDeep,
                border = BorderStroke(1.dp, Line),
                modifier = Modifier
                    .heightIn(min = 44.dp)
                    .clip(PillShape)
                    .clickable(enabled = !loading, onClick = onLoadMore),
            ) {
                Box(
                    modifier = Modifier
                        .heightIn(min = 44.dp)
                        .padding(horizontal = 24.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = if (loading) "Loading…" else "Load ${minOf(PHOTO_PAGE_SIZE, remaining)} more",
                        style = Typography.bodyMedium,
                        fontWeight = FontWeight.Medium,
                        color = if (loading) SlateSoft else Ink,
                    )
                }
            }
        }
    }
}

/* ─────────────── EMPTY RESULTS ─────────────── */

// Status-aware copy — ports the website's BibEmptyResult. (The web's notify
// email form is deliberately absent: the PhotoAlertCard is mobile's native
// equivalent of that intent.)
@Composable
private fun BibEmptyResult(
    bib: String,
    eventName: String,
    eventState: EventState,
    onClear: () -> Unit,
) {
    val (title, body) = when (eventState) {
        EventState.LIVE ->
            "Still uploading." to
                "Photographers are still working through this race — check back soon for $bib."
        EventState.PAST ->
            "This race has wrapped." to
                "Photos for $bib never landed in this archive. The wall's still here if you want to skim."
        else ->
            "Bib not found." to
                "All photos for this race have been uploaded — $bib isn't in there. Double-check the number, or skim the wall."
    }
    QpCard(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 24.dp),
        padding = 24.dp,
    ) {
        Kicker("Bib $bib · $eventName")
        Spacer(Modifier.height(12.dp))
        Text(title, style = Typography.titleLarge, color = Ink)
        Spacer(Modifier.height(8.dp))
        Text(body, style = Typography.bodyMedium, color = InkSoft)
        Spacer(Modifier.height(12.dp))
        Row(
            modifier = Modifier
                .heightIn(min = 48.dp)
                .clip(PillShape)
                .clickable(onClick = onClear),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            ArrowLabel("Or skim the full gallery →", color = Slate)
        }
    }
}

@Composable
private fun FaceEmptyResult(onClear: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 40.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Kicker("No matches yet")
        Spacer(Modifier.height(12.dp))
        Text(
            "We didn't find your face.",
            style = Typography.titleLarge,
            color = Ink,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "Try adding another selfie angle, or browse the wall while photos roll in.",
            style = Typography.bodyMedium,
            color = InkSoft,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(20.dp))
        PrimaryCta(text = "Browse the wall →", onClick = onClear)
    }
}

// Browse-all with zero photos (website GalleryEmptyResult): explain the
// timeline and offer the notify opt-in.
@Composable
private fun GalleryEmptyResult(alertCard: (@Composable () -> Unit)?) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 40.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Kicker("No photos yet")
        Spacer(Modifier.height(12.dp))
        Text(
            "Race photos aren't available yet.",
            style = Typography.titleLarge,
            color = Ink,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "Photographers upload within a few days of race day. Get notified the moment your photos land.",
            style = Typography.bodyMedium,
            color = InkSoft,
            textAlign = TextAlign.Center,
        )
        if (alertCard != null) {
            Spacer(Modifier.height(20.dp))
            alertCard()
        }
    }
}

/* ─────────────── LIVE + SKELETON ─────────────── */

/**
 * Live-arrival strip under the sticky search bar. Port of the website's
 * cockpit banner (`event-cockpit.tsx`): a count of photos that landed while
 * the runner was looking, or a manual refresh once the socket has stopped
 * healing itself.
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
                .clickable { if (giveUp) onRetry() else onJumpToTop() }
                .padding(horizontal = 24.dp, vertical = 12.dp),
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

@Composable
private fun PhotoGridSkeleton() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        repeat(2) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                repeat(2) {
                    LoadingSkeleton(
                        shape = MosaicTileShape,
                        modifier = Modifier
                            .weight(1f)
                            .aspectRatio(0.85f),
                    )
                }
            }
        }
    }
}
