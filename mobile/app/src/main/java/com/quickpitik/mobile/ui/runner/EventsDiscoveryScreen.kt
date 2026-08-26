package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
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
import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.*

// Date-filter chips, mirroring the website's date dropdown. Each (except ANY) maps
// to a single lifecycle bucket; selecting one collapses the segmented view into a
// flat list of just that bucket — same behaviour as /events on the web.
private enum class DateFilter(val label: String, val match: EventState?) {
    ANY("Any time", null),
    UPCOMING("Upcoming", EventState.UPCOMING),
    LIVE("Live now", EventState.LIVE),
    RECENT("Recent", EventState.OPEN),
    ARCHIVE("Archive", EventState.PAST),
}

// The four segment blocks rendered (in this order) when no filter is active, matching
// the web's SEGMENT_GROUPS. Live is the highlighted block.
private data class EventSegment(
    val title: String,
    val caption: String,
    val state: EventState,
    val highlight: Boolean,
)

private val SEGMENTS = listOf(
    EventSegment("Upcoming", "Save the date · Photos coming soon", EventState.UPCOMING, false),
    EventSegment("Live now", "Race day · Photos uploading", EventState.LIVE, true),
    EventSegment("Recent", "Last 30 days · Photos all in", EventState.OPEN, false),
    EventSegment("Archive", "Older races · Still searchable", EventState.PAST, false),
)

private const val FLAT_PAGE_SIZE = 12

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EventsDiscoveryScreen(
    viewModel: RunnerGalleryViewModel,
    savedEventsViewModel: SavedEventsViewModel,
    inboxViewModel: RunnerInboxViewModel,
    onEventSelected: (EventDto) -> Unit,
    onNavigateToOrders: () -> Unit,
    onNavigateToProfile: () -> Unit,
    onNavigateToSettings: () -> Unit,
    onOpenOrder: (String) -> Unit,
    onLogout: () -> Unit,
) {
    val eventsState by viewModel.eventsState.collectAsState()
    val savedIds by savedEventsViewModel.savedIds.collectAsState()

    // rememberSaveable: filters survive rotation (they still reset on
    // navigation away — URL-persisted filters are a web-only affordance).
    var query by rememberSaveable { mutableStateOf("") }
    var cityFilter by rememberSaveable { mutableStateOf("all") }
    var dateFilter by rememberSaveable { mutableStateOf(DateFilter.ANY) }

    val inboxMessages by inboxViewModel.messages.collectAsState()
    val inboxUnread by inboxViewModel.unreadCount.collectAsState()
    var showInbox by remember { mutableStateOf(false) }

    val context = LocalContext.current

    // Saved-events list is refreshed each time the browse screen is shown (the
    // runner is always authed here), keeping the heart state and the profile
    // race log in sync. Toggle feedback surfaces as a snackbar.
    val snackbarHostState = remember { SnackbarHostState() }
    val savedMessage by savedEventsViewModel.message.collectAsState()
    // Refetch on every screen entry — the VM's `init` only runs once at process
    // start, so an event the admin published mid-session would never appear
    // otherwise. The VM keeps the existing list visible during the refetch
    // (Loading state is only assigned when no prior Success exists).
    LaunchedEffect(Unit) { viewModel.fetchPublicEvents() }
    // Saved events + the runner inbox are RUNNER-role-gated server-side — a
    // photographer browsing in runner view (ViewMode) gets neither: the fetch
    // and the socket are skipped, the bell and hearts don't render. Web parity.
    val isTrueRunner = rememberIsTrueRunner()
    if (isTrueRunner) {
        LaunchedEffect(Unit) { savedEventsViewModel.refresh() }
    }
    // Inbox pushes live over /ws/me/runner/notifications, held only while this
    // screen is on-screen so a pocketed phone isn't keeping a socket open. The
    // socket refetches on every (re)open, which also covers the cold-start load.
    val lifecycleOwner = LocalLifecycleOwner.current
    if (isTrueRunner) DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> inboxViewModel.connect()
                Lifecycle.Event.ON_STOP -> inboxViewModel.disconnect()
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            inboxViewModel.disconnect()
        }
    }
    LaunchedEffect(savedMessage) {
        savedMessage?.let {
            snackbarHostState.showSnackbar(it)
            savedEventsViewModel.clearMessage()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
    Surface(modifier = Modifier.fillMaxSize(), color = Bone) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(top = 24.dp)
        ) {
            // ---- Sticky Header ----
            RunnerTopBar(
                kicker = "RACE PHOTOS · CEBU",
                onLogout = onLogout,
                trailingContent = {
                    if (isTrueRunner) {
                        RunnerInboxBell(
                            messageCount = inboxMessages.size,
                            unreadCount = inboxUnread,
                            onClick = { showInbox = true },
                        )
                    }
                }
            )

            Spacer(Modifier.height(24.dp))

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 24.dp)
            ) {
                // ---- Hero ----
                Text(
                    "Pick your race.",
                    style = Typography.displayLarge,
                    color = Ink
                )
            Text(
                "Find your photos.",
                style = Typography.displayLarge,
                color = Fresh
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "Open an event, then search by face or bib. Free to look — pay only for the ones you keep.",
                style = Typography.bodyMedium,
                color = InkSoft
            )

            Spacer(Modifier.height(20.dp))

            // ---- Content ----
            when (val state = eventsState) {
                is RunnerEventsState.Loading -> {
                    EventGridSkeleton()
                }

                is RunnerEventsState.Error -> {
                    ErrorView(
                        message = state.message,
                        onRetry = { viewModel.fetchPublicEvents() },
                        retryLabel = "Try again",
                        modifier = Modifier.fillMaxWidth(),
                    )
                }

                is RunnerEventsState.Success -> {
                    val events = state.events
                    val today = remember(events) { java.time.LocalDate.now() }
                    val withState = remember(events) { events.map { it to it.lifecycleState(today) } }
                    val cities = remember(events) {
                        events.map { extractCity(it.location) }.filter { it.isNotBlank() }.distinct().sorted()
                    }
                    val liveCount = withState.count { it.second == EventState.LIVE }
                    val trimmed = query.trim().lowercase()
                    val filtersActive = cityFilter != "all" || dateFilter != DateFilter.ANY || trimmed.isNotEmpty()

                    val filtered = withState.filter { (event, eventState) ->
                        if (cityFilter != "all" && extractCity(event.location) != cityFilter) return@filter false
                        dateFilter.match?.let { if (eventState != it) return@filter false }
                        if (trimmed.isNotEmpty()) {
                            val hay = "${event.name} ${event.location} ${event.date}".lowercase()
                            if (!hay.contains(trimmed)) return@filter false
                        }
                        true
                    }

                    // Live-today banner — jump nothing, just a presence cue like the web.
                    if (liveCount > 0) {
                        Surface(
                            shape = QpCardShape,
                            color = Fresh.copy(alpha = 0.10f),
                            border = BorderStroke(1.dp, Fresh.copy(alpha = 0.4f)),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                Box(
                                    modifier = Modifier.size(8.dp).clip(CircleShape).background(Fresh)
                                )
                                Text(
                                    "$liveCount race${if (liveCount == 1) "" else "s"} live today — photos uploading now",
                                    style = Typography.bodyMedium,
                                    fontWeight = FontWeight.Bold,
                                    color = FreshDeep
                                )
                            }
                        }
                        Spacer(Modifier.height(16.dp))
                    }

                    // ---- Filter strip ----
                    SearchField(query = query, onChange = { query = it })
                    Spacer(Modifier.height(10.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                        FilterDropdown(
                            label = if (cityFilter == "all") "All cities" else cityFilter,
                            options = listOf("all") + cities,
                            optionLabel = { if (it == "all") "All cities" else it },
                            onSelect = { cityFilter = it },
                            modifier = Modifier.weight(1f)
                        )
                        FilterDropdown(
                            label = dateFilter.label,
                            options = DateFilter.values().map { it.name },
                            optionLabel = { DateFilter.valueOf(it).label },
                            onSelect = { dateFilter = DateFilter.valueOf(it) },
                            modifier = Modifier.weight(1f)
                        )
                    }
                    if (filtersActive) {
                        Spacer(Modifier.height(8.dp))
                        Text(
                            "Clear filters ✕",
                            style = Typography.labelMedium,
                            color = Fresh,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.clickable {
                                query = ""; cityFilter = "all"; dateFilter = DateFilter.ANY
                            }
                        )
                    }

                    Spacer(Modifier.height(20.dp))

                    if (events.isEmpty()) {
                        EmptyState("No events yet.", "Check back soon — races appear here as photographers publish them.")
                    } else if (!filtersActive) {
                        // Segmented view (no filters): four blocks, empty ones skipped.
                        var rendered = 0
                        SEGMENTS.forEach { segment ->
                            val items = withState.filter { it.second == segment.state }
                            if (items.isNotEmpty()) {
                                if (rendered > 0) Spacer(Modifier.height(28.dp))
                                SegmentHeader(segment, items.size)
                                Spacer(Modifier.height(12.dp))
                                EventGrid(items, savedIds, onEventSelected, savedEventsViewModel::toggle)
                                rendered++
                            }
                        }
                    } else {
                        // Flat filtered view with load-more, like the web's FlatResults.
                        var visibleCount by rememberSaveable(query, cityFilter, dateFilter) { mutableStateOf(FLAT_PAGE_SIZE) }
                        val cityLabel = if (cityFilter == "all") "All cities" else cityFilter
                        Text(
                            "${filtered.size} of ${withState.size} ${if (withState.size == 1) "race" else "races"}",
                            style = Typography.labelMedium,
                            color = Slate,
                            fontWeight = FontWeight.Bold
                        )
                        Text("$cityLabel · ${dateFilter.label}", style = Typography.bodySmall, color = SlateSoft)
                        Spacer(Modifier.height(12.dp))
                        if (filtered.isEmpty()) {
                            EmptyState(
                                "No races match your filters.",
                                "Try adjusting the date, city, or search above."
                            )
                        } else {
                            EventGrid(filtered.take(visibleCount), savedIds, onEventSelected, savedEventsViewModel::toggle)
                            if (filtered.size > visibleCount) {
                                Spacer(Modifier.height(16.dp))
                                GhostCta(
                                    text = "LOAD MORE",
                                    onClick = { visibleCount += FLAT_PAGE_SIZE },
                                    modifier = Modifier.fillMaxWidth(),
                                )
                            }
                        }
                    }
                }
            }

            Spacer(Modifier.height(24.dp))
        }
        }
    }
        SnackbarHost(
            hostState = snackbarHostState,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .navigationBarsPadding()
                .padding(16.dp)
        )
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

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SearchField(query: String, onChange: (String) -> Unit) {
    TextField(
        value = query,
        onValueChange = onChange,
        placeholder = { Text("Search by event name…", color = SlateSoft) },
        leadingIcon = { Icon(Icons.Default.Search, contentDescription = null, tint = Slate) },
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

@Composable
private fun FilterDropdown(
    label: String,
    options: List<String>,
    optionLabel: (String) -> String,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }
    Box(modifier = modifier) {
        Surface(
            onClick = { expanded = true },
            shape = FieldShape,
            color = BoneDeep,
            border = BorderStroke(1.dp, Line),
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    label,
                    style = Typography.bodyMedium,
                    color = Ink,
                    maxLines = 1,
                    modifier = Modifier.weight(1f, fill = false)
                )
                Icon(Icons.Default.ArrowDropDown, contentDescription = "Expand", tint = Slate)
            }
        }
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
            modifier = Modifier.background(Bone)
        ) {
            options.forEach { option ->
                DropdownMenuItem(
                    text = { Text(optionLabel(option), color = Ink) },
                    onClick = { onSelect(option); expanded = false }
                )
            }
        }
    }
}

@Composable
private fun SegmentHeader(segment: EventSegment, count: Int) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (segment.highlight) {
                    Box(modifier = Modifier.size(8.dp).clip(CircleShape).background(Fresh))
                }
                Text(
                    segment.title.uppercase(),
                    style = Typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = if (segment.highlight) Fresh else Slate
                )
            }
            Text(
                "$count ${if (count == 1) "race" else "races"}",
                style = Typography.labelSmall,
                color = SlateSoft
            )
        }
        Text(segment.caption, style = Typography.bodySmall, color = SlateSoft)
        Spacer(Modifier.height(8.dp))
        Divider(color = Line)
    }
}

// 2-column grid via chunked rows (avoids nesting a lazy grid inside the scrolling Column).
@Composable
private fun EventGrid(
    items: List<Pair<EventDto, EventState>>,
    savedIds: Set<String>,
    onSelect: (EventDto) -> Unit,
    onToggleSave: (EventDto) -> Unit,
) {
    items.chunked(2).forEach { row ->
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            row.forEach { (event, eventState) ->
                Box(modifier = Modifier.weight(1f)) {
                    EventTile(event, eventState, event.id in savedIds, onSelect, onToggleSave)
                }
            }
            if (row.size == 1) Spacer(Modifier.weight(1f))
        }
        Spacer(Modifier.height(12.dp))
    }
}

@Composable
private fun EventTile(
    event: EventDto,
    state: EventState,
    saved: Boolean,
    onSelect: (EventDto) -> Unit,
    onToggleSave: (EventDto) -> Unit,
) {
    // Upcoming events are non-interactive on the web (no gallery yet) — match that.
    val interactive = state != EventState.UPCOMING
    Card(
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        border = BorderStroke(1.dp, Line),
        modifier = Modifier
            .fillMaxWidth()
            .then(if (interactive) Modifier.clickable { onSelect(event) } else Modifier)
    ) {
        Column {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(4f / 3f)
                    .background(Ink)
            ) {
                if (!event.bannerUrl.isNullOrEmpty()) {
                    AsyncImage(
                        model = RetrofitClient.resolveImageUrl(event.bannerUrl),
                        contentDescription = event.name,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop
                    )
                } else {
                    Text(
                        "BANNER · SOON",
                        style = Typography.labelSmall,
                        color = Bone.copy(alpha = 0.55f),
                        modifier = Modifier.align(Alignment.Center)
                    )
                }
                StatusChip(
                    state = state,
                    modifier = Modifier.align(Alignment.TopStart).padding(10.dp)
                )
                // Saved events are RUNNER-gated server-side — hidden for a
                // photographer browsing in runner view (see rememberIsTrueRunner).
                if (rememberIsTrueRunner()) {
                    SaveButton(
                        saved = saved,
                        onToggle = { onToggleSave(event) },
                        modifier = Modifier.align(Alignment.TopEnd).padding(6.dp)
                    )
                }
            }
            Column(modifier = Modifier.padding(14.dp)) {
                Text(
                    "${eventDateLabel(event.date)} · ${extractCity(event.location).uppercase()}",
                    style = Typography.labelSmall,
                    color = Slate
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    event.name,
                    style = Typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Ink,
                    maxLines = 2
                )
                Text(event.location, style = Typography.bodySmall, color = SlateSoft, maxLines = 1)
                Spacer(Modifier.height(10.dp))
                ArrowLabel(
                    if (interactive) "Open →" else "Opens on race day",
                    color = if (interactive) Fresh else SlateSoft,
                    style = Typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                )
            }
        }
    }
}

// Bookmark toggle — the website uses a bookmark glyph; mobile uses a heart
// (FavoriteBorder / Favorite) because material-icons-extended (where Bookmark
// lives) isn't on the classpath, only the core set. Same save/un-save intent.
@Composable
private fun SaveButton(saved: Boolean, onToggle: () -> Unit, modifier: Modifier = Modifier) {
    Surface(
        shape = CircleShape,
        color = Ink.copy(alpha = 0.55f),
        modifier = modifier.size(34.dp)
    ) {
        IconButton(onClick = onToggle) {
            Icon(
                imageVector = if (saved) Icons.Default.Favorite else Icons.Default.FavoriteBorder,
                contentDescription = if (saved) "Remove from saved" else "Save event",
                tint = if (saved) Fresh else Bone,
                modifier = Modifier.size(18.dp)
            )
        }
    }
}

@Composable
private fun StatusChip(state: EventState, modifier: Modifier = Modifier) {
    val (label, accent) = when (state) {
        EventState.LIVE -> "PHOTOS UPLOADING" to Fresh
        EventState.UPCOMING -> "SAVE THE DATE" to Bone
        EventState.OPEN -> "PHOTOS READY" to Bone
        EventState.PAST -> "ARCHIVE" to Bone.copy(alpha = 0.7f)
    }
    Surface(
        shape = RoundedCornerShape(percent = 100),
        color = Ink.copy(alpha = 0.55f),
        modifier = modifier
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Box(modifier = Modifier.size(6.dp).clip(CircleShape).background(accent))
            Text(
                label,
                style = Typography.labelSmall,
                fontWeight = FontWeight.Bold,
                color = accent,
            )
        }
    }
}

@Composable
private fun EmptyState(title: String, body: String) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(vertical = 40.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(title, style = Typography.titleMedium, fontWeight = FontWeight.Bold, color = Ink)
        Spacer(Modifier.height(6.dp))
        Text(body, style = Typography.bodyMedium, color = SlateSoft, textAlign = TextAlign.Center)
    }
}

// 2×3 skeleton grid that mirrors the EventTile footprint (4:3 banner + footer ≈ aspect 0.75).
// Replaces the centered spinner so the loading state hints at the final layout.
@Composable
private fun EventGridSkeleton() {
    Column(modifier = Modifier.fillMaxWidth()) {
        repeat(3) { rowIndex ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                repeat(2) {
                    LoadingSkeleton(
                        shape = QpCardShape,
                        modifier = Modifier
                            .weight(1f)
                            .aspectRatio(0.75f),
                    )
                }
            }
            if (rowIndex < 2) Spacer(Modifier.height(12.dp))
        }
    }
}
