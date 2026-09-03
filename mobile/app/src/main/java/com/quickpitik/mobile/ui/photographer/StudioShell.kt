package com.quickpitik.mobile.ui.photographer

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddCircle
import androidx.compose.material.icons.filled.ExitToApp
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.lightColorScheme
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.PhotographerMessageDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.BrandLogo
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.NumeralStyle
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.Typography
import com.quickpitik.mobile.ui.theme.WarningOrange

// ─────────────────────────────────────────────────────────────────────────────
// Studio shell — the chrome shared by every photographer ("studio/*") route.
//
// Extracted from DashboardScreen.kt when the five photographer tabs became real
// NavHost routes (2026-08-26). Before that, one 1774-line composable owned the
// Scaffold, a remember{} tab int, two sub-surface booleans, and the bottom nav
// — so system back exited the app from every tab and none of it survived
// process death. The tab CONTENT stayed where it was; only the chrome moved.
// ─────────────────────────────────────────────────────────────────────────────

/** The five bottom-nav tab routes, in display order. Sub-surfaces (share,
 *  profile preview) are deliberately absent — they render fullscreen. */
val STUDIO_TAB_ROUTES = listOf(
    "studio/home", "studio/capture", "studio/events", "studio/earnings", "studio/settings",
)

/**
 * The nested Material theme the old dashboard applied — kept identical so the
 * studio surfaces don't shift colors in the move. Wraps every studio route,
 * tabs and fullscreen sub-surfaces alike.
 */
@Composable
fun StudioTheme(content: @Composable () -> Unit) {
    // Explicitly lock the photographer studio into the warm cream theme.
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Fresh,
            onPrimary = Color.White,
            background = Bone,
            onBackground = Ink,
            surface = BoneDeep,
            onSurface = Ink,
            outline = Line
        ),
        // Carry the Quiet Studio type scale into the nested theme — otherwise
        // the studio reverts to the system font (Material default).
        typography = Typography,
        content = content,
    )
}

/**
 * Chrome for one TAB route: top bar + upload banner + the inbox dialog. The
 * bottom nav is NOT here — it lives in MainActivity's single Scaffold slot,
 * beside the runner nav, driven by the current route.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StudioTabScaffold(
    viewModel: PhotographerDashboardViewModel,
    onLogout: () -> Unit,
    onPreviewProfile: () -> Unit,
    onSwitchToRunner: () -> Unit,
    content: @Composable () -> Unit,
) {
    val messages by viewModel.messages.collectAsState()
    val brandSettings by viewModel.brandSettings.collectAsState()
    // Drives the cross-tab GlobalUploadBanner — surfaces sync progress so the
    // photographer doesn't lose track of in-flight uploads after dismissing
    // the import sheet. The Sync queue card on the Capture tab remains the
    // detailed view; this is the persistent ambient status.
    val queueStats by viewModel.queueStats.collectAsState()
    val watchState by viewModel.shutterWatchState.collectAsState()
    val verificationState by viewModel.verificationState.collectAsState()
    val currentStatus = (verificationState as? VerificationUiState.Success)?.verification?.status?.lowercase() ?: "incomplete"
    val isApproved = currentStatus == "approved"
    val isPending = currentStatus == "pending"
    val unreadCount = remember(messages) { messages.count { it.readAt == null } }
    var showNotifDialog by remember { mutableStateOf(false) }
    // Switching to runner view pops the studio graph, which clears this VM —
    // and with it a live shutter watch. Correct on purpose (same teardown as
    // logout), but it must never be an accidental one-tap mid-shoot.
    var confirmSwitch by remember { mutableStateOf(false) }
    val tetherLive = watchState is ShutterWatchState.Starting ||
        watchState is ShutterWatchState.Watching
    val resolvedAvatarUrl = RetrofitClient.resolveImageUrl(brandSettings?.avatarUrl)

    StudioTheme {
        if (showNotifDialog) {
            NotificationsInboxDialog(
                messages = messages,
                onDismiss = { showNotifDialog = false },
                onMarkRead = { id -> viewModel.markMessageAsRead(id) },
                onMarkAllRead = { viewModel.markAllMessagesAsRead() },
                onRemove = { id -> viewModel.removeMessage(id) },
            )
        }
        if (confirmSwitch) {
            AlertDialog(
                onDismissRequest = { confirmSwitch = false },
                title = {
                    Text(
                        text = "Stop capturing and switch?",
                        color = Ink,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                    )
                },
                text = {
                    Text(
                        text = "Your camera is capturing right now. Switching to the runner view stops the tether session — photos already queued keep uploading in the background.",
                        color = Slate,
                        fontSize = 14.sp,
                        lineHeight = 20.sp,
                    )
                },
                confirmButton = {
                    Button(
                        onClick = {
                            confirmSwitch = false
                            onSwitchToRunner()
                        },
                        shape = PillShape,
                        colors = ButtonDefaults.buttonColors(containerColor = Ink),
                        modifier = Modifier.height(40.dp),
                    ) {
                        Text(
                            text = "SWITCH",
                            color = Bone,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.5.sp,
                        )
                    }
                },
                dismissButton = {
                    OutlinedButton(
                        onClick = { confirmSwitch = false },
                        shape = PillShape,
                        border = BorderStroke(1.dp, Ink),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink),
                        modifier = Modifier.height(40.dp),
                    ) {
                        Text(
                            text = "KEEP CAPTURING",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.5.sp,
                        )
                    }
                },
                containerColor = Bone,
                shape = QpCardShape,
            )
        }
        Scaffold(
            containerColor = Bone,
            topBar = {
                PhotographerTopBar(
                    avatarUrl = resolvedAvatarUrl,
                    unreadCount = unreadCount,
                    isApproved = isApproved,
                    isPending = isPending,
                    onOpenNotifications = { showNotifDialog = true },
                    onPreviewProfile = onPreviewProfile,
                    onSwitchToRunner = {
                        if (tetherLive) confirmSwitch = true else onSwitchToRunner()
                    },
                    onLogout = onLogout,
                )
            },
        ) { paddingValues ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Bone)
                    .padding(paddingValues)
            ) {
                GlobalUploadBanner(queueStats = queueStats)
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                ) {
                    content()
                }
            }
        }
    }
}

/**
 * Inbox push channel (/ws/me/photographer/notifications) — an admin approving
 * a verification reaches the bell without a tab tap. Mounted ONCE from
 * MainActivity for the whole studio session, not per tab route: per-tab
 * placement would close/reopen the socket on every tab switch, and each
 * reopen's WsState.Open collector refires fetchMessages — a fetch storm.
 * Held only while the app is foregrounded: the tether foreground service is
 * the one path meant to survive backgrounding, and the bell must not extend it.
 */
@Composable
fun StudioInboxLifecycle(viewModel: PhotographerDashboardViewModel) {
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> viewModel.connectInbox()
                Lifecycle.Event.ON_STOP -> viewModel.disconnectInbox()
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            viewModel.disconnectInbox()
        }
    }
}

// Floating-pill bottom nav — Quiet Studio'd port of a glassmorphic source design.
// Flat-with-borders: Bone fill, 1dp Line outline, no blur / no shadow / no Fresh
// (Fresh stays reserved for the screen's primary CTA — the Settings dot is the
// one tolerated micro-accent, matching the website's notification badges).
// Selection is derived from the CURRENT ROUTE — the nav has no state of its own.
@Composable
fun PhotographerFloatingBottomNav(
    currentRoute: String?,
    showSettingsBadge: Boolean,
    onNavigate: (String) -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .windowInsetsPadding(WindowInsets.navigationBars)
            .padding(horizontal = 12.dp, vertical = 12.dp),
        contentAlignment = Alignment.Center,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(PillShape)
                .background(Bone)
                .border(BorderStroke(1.dp, Line), PillShape)
                .padding(6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            StudioNavItem(
                icon = Icons.Default.Home,
                label = "Home",
                selected = currentRoute == "studio/home",
                onClick = { onNavigate("studio/home") },
                modifier = Modifier.weight(1f),
            )
            StudioNavItem(
                icon = Icons.Default.AddCircle,
                label = "Capture",
                selected = currentRoute == "studio/capture",
                onClick = { onNavigate("studio/capture") },
                modifier = Modifier.weight(1f),
            )
            StudioNavItem(
                icon = Icons.Default.List,
                label = "Events",
                selected = currentRoute == "studio/events",
                onClick = { onNavigate("studio/events") },
                modifier = Modifier.weight(1f),
            )
            StudioNavItem(
                icon = Icons.Default.ShoppingCart,
                label = "Earnings",
                selected = currentRoute == "studio/earnings",
                onClick = { onNavigate("studio/earnings") },
                modifier = Modifier.weight(1f),
            )
            StudioNavItem(
                icon = Icons.Default.Settings,
                label = "Settings",
                selected = currentRoute == "studio/settings",
                onClick = { onNavigate("studio/settings") },
                modifier = Modifier.weight(1f),
                badge = showSettingsBadge,
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun StudioNavItem(
    icon: ImageVector,
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    badge: Boolean = false,
) {
    val bg by animateColorAsState(
        targetValue = if (selected) Ink else Color.Transparent,
        animationSpec = tween(180),
        label = "navItemBg",
    )
    val tint by animateColorAsState(
        targetValue = if (selected) Bone else Slate,
        animationSpec = tween(180),
        label = "navItemTint",
    )
    val scale by animateFloatAsState(
        targetValue = if (selected) 1.0f else 0.94f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessMediumLow,
        ),
        label = "navItemScale",
    )
    Column(
        modifier = modifier
            .heightIn(min = 56.dp)
            .clip(PillShape)
            .background(bg)
            .clickable(onClick = onClick)
            .graphicsLayer {
                scaleX = scale
                scaleY = scale
            }
            .padding(vertical = 8.dp, horizontal = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        if (badge) {
            BadgedBox(
                badge = {
                    Badge(
                        containerColor = Fresh,
                        modifier = Modifier.size(6.dp),
                    )
                },
            ) {
                Icon(icon, contentDescription = label, tint = tint, modifier = Modifier.size(22.dp))
            }
        } else {
            Icon(icon, contentDescription = label, tint = tint, modifier = Modifier.size(22.dp))
        }
        Spacer(Modifier.height(2.dp))
        Text(
            text = label,
            color = tint,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            maxLines = 1,
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PhotographerTopBar(
    avatarUrl: String?,
    unreadCount: Int,
    isApproved: Boolean = true,
    isPending: Boolean = false,
    onOpenNotifications: () -> Unit,
    onPreviewProfile: () -> Unit,
    onSwitchToRunner: () -> Unit,
    onLogout: () -> Unit,
) {
    var showAvatarMenu by remember { mutableStateOf(false) }
    var showLogoutConfirm by remember { mutableStateOf(false) }
    var showSetupAlert by remember { mutableStateOf(false) }

    if (showSetupAlert) {
        StudioSetupRequiredDialog(
            isPending = isPending,
            onDismiss = { showSetupAlert = false },
        )
    }
    TopAppBar(
        title = {
            Column {
                BrandLogo(compact = true)
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "STUDIO",
                    style = Typography.labelMedium,
                    color = Slate,
                )
            }
        },
        actions = {
            IconButton(onClick = onOpenNotifications) {
                Box {
                    Icon(
                        imageVector = Icons.Default.Notifications,
                        contentDescription = "Notifications",
                        tint = if (unreadCount > 0) Fresh else SlateSoft,
                    )
                    if (unreadCount > 0) {
                        Box(
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .size(14.dp)
                                .clip(CircleShape)
                                .background(ErrorRed)
                                .border(2.dp, Bone, CircleShape),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(
                                text = if (unreadCount > 9) "9+" else unreadCount.toString(),
                                color = Color.White,
                                fontSize = 8.sp,
                                fontWeight = FontWeight.Bold,
                            )
                        }
                    }
                }
            }
            Box {
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .clip(CircleShape)
                        .background(BoneDeep)
                        .border(1.dp, Line, CircleShape)
                        .clickable { showAvatarMenu = true },
                    contentAlignment = Alignment.Center,
                ) {
                    if (!avatarUrl.isNullOrBlank()) {
                        AsyncImage(
                            model = avatarUrl,
                            contentDescription = "Profile menu",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize().clip(CircleShape),
                        )
                    } else {
                        Text(
                            text = "P",
                            color = Ink,
                            style = Typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                        )
                    }
                }
                DropdownMenu(
                    expanded = showAvatarMenu,
                    onDismissRequest = { showAvatarMenu = false },
                    modifier = Modifier.background(BoneDeep),
                ) {
                    DropdownMenuItem(
                        text = { Text("Preview public profile", color = Ink, fontSize = 13.sp) },
                        leadingIcon = {
                            Icon(
                                imageVector = Icons.Default.Person,
                                contentDescription = null,
                                tint = Ink,
                                modifier = Modifier.size(18.dp),
                            )
                        },
                        onClick = {
                            showAvatarMenu = false
                            if (!isApproved) {
                                showSetupAlert = true
                            } else {
                                onPreviewProfile()
                            }
                        },
                    )
                    DropdownMenuItem(
                        text = { Text("Switch to runner", color = Ink, fontSize = 13.sp) },
                        leadingIcon = {
                            Icon(
                                imageVector = Icons.Default.Refresh,
                                contentDescription = null,
                                tint = Ink,
                                modifier = Modifier.size(18.dp),
                            )
                        },
                        onClick = {
                            showAvatarMenu = false
                            onSwitchToRunner()
                        },
                    )
                    Divider(color = Line)
                    DropdownMenuItem(
                        text = { Text("Sign out", color = ErrorRed, fontSize = 13.sp) },
                        leadingIcon = {
                            Icon(
                                imageVector = Icons.Default.ExitToApp,
                                contentDescription = null,
                                tint = ErrorRed,
                                modifier = Modifier.size(18.dp),
                            )
                        },
                        onClick = {
                            showAvatarMenu = false
                            showLogoutConfirm = true
                        },
                    )
                }
            }
            Spacer(modifier = Modifier.width(4.dp))
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = Bone,
            scrolledContainerColor = Bone,
            navigationIconContentColor = Ink,
            titleContentColor = Ink,
            actionIconContentColor = Ink,
        ),
    )

    if (showLogoutConfirm) {
        AlertDialog(
            onDismissRequest = { showLogoutConfirm = false },
            containerColor = Bone,
            title = {
                Text(
                    text = "Sign out of QuickPitik?",
                    style = Typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Ink,
                )
            },
            text = {
                Text(
                    text = "You will need to sign in again to access your studio, upload queues, and earnings.",
                    style = Typography.bodyMedium,
                    color = Slate,
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showLogoutConfirm = false
                        onLogout()
                    }
                ) {
                    Text("SIGN OUT", color = ErrorRed, fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = {
                TextButton(onClick = { showLogoutConfirm = false }) {
                    Text("CANCEL", color = Slate, fontWeight = FontWeight.Bold)
                }
            }
        )
    }
}

/**
 * Cross-tab ambient progress strip for the photo-upload queue.
 *
 * Sits between the TopAppBar and the tab content. Visible whenever the queue
 * has anything UPLOADING or QUEUED — auto-hides as soon as both counts drain.
 * Pairs with the detailed Sync queue card on the Capture tab; the banner is
 * the "don't close the app" signal, the card is the breakdown.
 *
 * Quiet Studio: BoneDeep fill, Slate progress, no Fresh accent (the active
 * tab's PrimaryCta keeps the single Fresh allowance). Slide + fade enter/exit
 * — never fade alone per the Mobile Design motion rule.
 */
@Composable
private fun GlobalUploadBanner(queueStats: QueueStats) {
    val inFlight = queueStats.uploadingCount + queueStats.queuedCount
    AnimatedVisibility(
        visible = inFlight > 0,
        enter = slideInVertically(initialOffsetY = { -it }) + fadeIn(),
        exit = slideOutVertically(targetOffsetY = { -it }) + fadeOut(),
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(BoneDeep)
                    .padding(horizontal = 20.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                CircularProgressIndicator(
                    color = Slate,
                    strokeWidth = 2.dp,
                    modifier = Modifier.size(14.dp),
                )
                Spacer(modifier = Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (queueStats.uploadingCount > 0) "Uploading photos" else "Photos queued",
                        color = Ink,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold,
                    )
                    val detail = buildString {
                        if (queueStats.uploadingCount > 0) append("${queueStats.uploadingCount} in progress")
                        if (queueStats.queuedCount > 0) {
                            if (isNotEmpty()) append(" · ")
                            append("${queueStats.queuedCount} queued")
                        }
                    }
                    if (detail.isNotBlank()) {
                        Spacer(modifier = Modifier.height(2.dp))
                        Text(
                            text = detail,
                            color = SlateSoft,
                            style = NumeralStyle.copy(fontSize = 11.sp),
                        )
                    }
                }
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = inFlight.toString(),
                    color = Ink,
                    style = NumeralStyle.copy(fontSize = 16.sp),
                    fontWeight = FontWeight.SemiBold,
                )
            }
            LinearProgressIndicator(
                progress = queueStats.progress,
                color = Slate,
                trackColor = Line,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(2.dp),
            )
            Divider(color = Line)
        }
    }
}

// A ModalBottomSheet, not an AlertDialog — consistent with the runner inbox,
// whose header comment already states the rule: a scrolling message list is
// exactly the case a dialog handles badly on a phone.
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun NotificationsInboxDialog(
    messages: List<PhotographerMessageDto>,
    onDismiss: () -> Unit,
    onMarkRead: (String) -> Unit,
    onMarkAllRead: () -> Unit,
    onRemove: (String) -> Unit,
) {
    var removeTarget by remember { mutableStateOf<PhotographerMessageDto?>(null) }
    val unreadCount = messages.count { it.readAt == null }
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Bone,
        dragHandle = null,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp)
                .padding(top = 20.dp, bottom = 28.dp),
        ) {
            Text(
                text = "Inbox",
                color = Ink,
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(12.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Kicker(
                    text = if (unreadCount > 0) "$unreadCount unread · ${messages.size} total"
                           else "${messages.size} total",
                    color = SlateSoft,
                )
                if (unreadCount > 0) {
                    Text(
                        text = "MARK ALL READ",
                        color = Slate,
                        style = Typography.labelMedium,
                        modifier = Modifier
                            .clickable { onMarkAllRead() }
                            .padding(horizontal = 4.dp, vertical = 4.dp),
                    )
                }
            }
            if (messages.isEmpty()) {
                Text(
                    text = "No messages yet. Admin actions on your account will land here.",
                    color = SlateSoft,
                    fontSize = 14.sp,
                    textAlign = TextAlign.Center,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 32.dp),
                )
            } else {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 480.dp)
                        .verticalScroll(rememberScrollState()),
                ) {
                    messages.forEachIndexed { index, msg ->
                        if (index > 0) Divider(color = Line)
                        InboxRow(
                            message = msg,
                            onMarkRead = { onMarkRead(msg.id) },
                            onRemove = { removeTarget = msg },
                        )
                    }
                }
            }
        }
    }

    val target = removeTarget
    if (target != null) {
        AlertDialog(
            onDismissRequest = { removeTarget = null },
            title = {
                Text(
                    text = "Remove this notification?",
                    color = Ink,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                )
            },
            text = {
                Text(
                    text = "This message will be cleared from your inbox. Admin still has the underlying record on the decision log.",
                    color = Slate,
                    fontSize = 14.sp,
                    lineHeight = 20.sp,
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        onRemove(target.id)
                        removeTarget = null
                    },
                    shape = PillShape,
                    colors = ButtonDefaults.buttonColors(containerColor = ErrorRed),
                    modifier = Modifier.height(40.dp),
                ) {
                    Text(
                        text = "REMOVE",
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.5.sp,
                    )
                }
            },
            dismissButton = {
                OutlinedButton(
                    onClick = { removeTarget = null },
                    shape = PillShape,
                    border = BorderStroke(1.dp, Ink),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink),
                    modifier = Modifier.height(40.dp),
                ) {
                    Text(
                        text = "CANCEL",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.5.sp,
                    )
                }
            },
            containerColor = Bone,
            shape = QpCardShape,
        )
    }
}

@Composable
private fun InboxRow(
    message: PhotographerMessageDto,
    onMarkRead: () -> Unit,
    onRemove: () -> Unit,
) {
    val isUnread = message.readAt == null
    val toneColor = when (message.kind) {
        "verification_approved", "unsuspended", "dispute_resolved",
        "payout_approved", "payout_paid", "payout_report_resolved" -> Fresh
        "verification_rejected", "verification_reset",
        "dispute_escalated", "payout_held", "dispute_denied" -> WarningOrange
        "suspended" -> ErrorRed
        "admin_message", "payout_report_acknowledged" -> Ink
        // force_edit is deliberately absent: the backend removed that admin
        // action end-to-end, so the kind can never be emitted. Anything else
        // new from the backend lands on Slate rather than a wrong colour.
        else -> Slate
    }
    val title = message.title?.trim().takeUnless { it.isNullOrEmpty() }
        ?: messageKindLabel(message.kind)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                if (isUnread) BoneDeep.copy(alpha = 0.4f) else Color.Transparent,
            )
            .clickable { if (isUnread) onMarkRead() }
            .padding(horizontal = 8.dp, vertical = 14.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (isUnread) {
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .clip(CircleShape)
                            .background(Fresh),
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                }
                Text(
                    text = messageKindLabel(message.kind).uppercase(),
                    style = Typography.labelMedium,
                    color = toneColor,
                )
            }
            Text(
                text = formatInboxDate(message.createdAt),
                style = Typography.labelMedium,
                color = SlateSoft,
            )
        }
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = title,
            color = if (isUnread) Ink else InkSoft,
            fontSize = 16.sp,
            fontWeight = if (isUnread) FontWeight.SemiBold else FontWeight.Normal,
            lineHeight = 22.sp,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = message.body,
            color = Slate,
            fontSize = 14.sp,
            lineHeight = 20.sp,
        )
        Spacer(modifier = Modifier.height(10.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End,
        ) {
            Text(
                text = "REMOVE",
                color = SlateSoft,
                style = Typography.labelMedium,
                modifier = Modifier
                    .clickable { onRemove() }
                    .padding(horizontal = 6.dp, vertical = 4.dp),
            )
        }
    }
}

private fun messageKindLabel(kind: String): String = when (kind) {
    "verification_approved" -> "Verification approved"
    "verification_rejected" -> "Verification needs changes"
    "verification_reset" -> "Verification reset"
    "suspended" -> "Account suspended"
    "unsuspended" -> "Account reinstated"
    "force_edit" -> "Profile updated by admin"
    "dispute_resolved" -> "Refund resolved"
    "dispute_denied" -> "Refund denied"
    "dispute_escalated" -> "Refund escalated"
    "payout_approved" -> "Payout approved"
    "payout_held" -> "Payout held"
    "payout_paid" -> "Payout paid"
    "payout_report_acknowledged" -> "Report acknowledged"
    "payout_report_resolved" -> "Report resolved"
    "admin_message" -> "Message from admin"
    else -> kind.replace('_', ' ').replaceFirstChar { it.uppercase() }
}

// "AUG 14, 2026" — the year matters on messages that live for months
// (web formatLongDate parity, same as the runner inbox).
private fun formatInboxDate(iso: String): String {
    return try {
        val parts = iso.substring(0, 10).split("-")
        val months = listOf(
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
            "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
        )
        "${months[parts[1].toInt() - 1]} ${parts[2].toInt()}, ${parts[0]}"
    } catch (e: Exception) { iso }
}

@Composable
fun StudioSetupRequiredDialog(
    isPending: Boolean = false,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            Button(
                onClick = onDismiss,
                colors = ButtonDefaults.buttonColors(containerColor = Fresh),
            ) {
                Text("OK", color = Color.White)
            }
        },
        title = {
            Text(
                text = if (isPending) "Verification Review Pending" else "Onboarding Setup Required",
                fontWeight = FontWeight.Bold,
                color = Ink,
            )
        },
        text = {
            Text(
                text = if (isPending) {
                    "Your professional studio setup is currently being reviewed by an administrator. Please wait for approval before covering events."
                } else {
                    "Your studio setup is not approved. Please complete the setup on the Settings tab and wait for administrator approval."
                },
                color = Ink,
            )
        },
        containerColor = BoneDeep,
    )
}

