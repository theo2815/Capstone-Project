package com.quickpitik.mobile

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.runtime.collectAsState
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavBackStackEntry
import androidx.navigation.NavController
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.navigation
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.quickpitik.mobile.data.local.SessionEvents
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.ViewMode
import com.quickpitik.mobile.data.local.isPhotographerRole
import com.quickpitik.mobile.ui.auth.AuthViewModel
import com.quickpitik.mobile.ui.auth.ForgotPasswordScreen
import com.quickpitik.mobile.ui.auth.LoginScreen
import com.quickpitik.mobile.ui.auth.RegisterScreen
import com.quickpitik.mobile.ui.photographer.EventsState
import com.quickpitik.mobile.ui.photographer.PhotographerCaptureScreen
import com.quickpitik.mobile.ui.photographer.PhotographerDashboardViewModel
import com.quickpitik.mobile.ui.photographer.PhotographerEarningsScreen
import com.quickpitik.mobile.ui.photographer.PhotographerEventShareScreen
import com.quickpitik.mobile.ui.photographer.PhotographerEventsScreen
import com.quickpitik.mobile.ui.photographer.PhotographerFloatingBottomNav
import com.quickpitik.mobile.ui.photographer.PhotographerOverviewScreen
import com.quickpitik.mobile.ui.photographer.PhotographerPublicProfileScreen
import com.quickpitik.mobile.ui.photographer.PhotographerSettingsScreen
import com.quickpitik.mobile.ui.photographer.PublicPhotographerViewModel
import com.quickpitik.mobile.ui.photographer.STUDIO_TAB_ROUTES
import com.quickpitik.mobile.ui.photographer.StudioInboxLifecycle
import com.quickpitik.mobile.ui.photographer.StudioTabScaffold
import com.quickpitik.mobile.ui.photographer.StudioTheme
import com.quickpitik.mobile.ui.photographer.VerificationUiState
import com.quickpitik.mobile.ui.runner.EventsDiscoveryScreen
import com.quickpitik.mobile.ui.runner.FloatingCart
import com.quickpitik.mobile.ui.runner.RunnerGalleryScreen
import com.quickpitik.mobile.ui.runner.RunnerGalleryViewModel
import com.quickpitik.mobile.ui.runner.RunnerInboxViewModel
import com.quickpitik.mobile.ui.runner.SavedEventsViewModel
import com.quickpitik.mobile.ui.runner.CartViewModel
import com.quickpitik.mobile.ui.runner.OrderReturnScreen
import com.quickpitik.mobile.ui.runner.OrdersScreen
import com.quickpitik.mobile.ui.runner.ProfileViewModel
import com.quickpitik.mobile.ui.runner.ProfileScreen
import com.quickpitik.mobile.ui.runner.AccountSettingsScreen
import com.quickpitik.mobile.ui.theme.QuickPitikMobileTheme
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.ui.Alignment
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.navigation.compose.currentBackStackEntryAsState

// Runner-owned route PATTERNS (as the back stack reports them). Used by the
// role guard and, minus the receipt route, by the bottom-nav gate.
// "photographer/{handle}" and the auth routes belong to NEITHER role set —
// they are shared surfaces.
private val RUNNER_ROUTES = setOf(
    "events", "gallery", "profile", "settings",
    "orders?orderId={orderId}", "orders/return/{orderId}",
)

// Where the runner bottom nav shows: the tab surfaces + the orders list, but
// NOT the modal PayMongo receipt ("orders/return/{orderId}").
private val RUNNER_NAV_ROUTES = setOf(
    "events", "gallery", "profile", "settings", "orders?orderId={orderId}",
)

class MainActivity : ComponentActivity() {
    // Latest deep-link URI from a quickpitik:// intent. Compose observes this
    // via a LaunchedEffect and routes the user to OrdersScreen (return) or
    // back to the cart sheet (cancel) once the auth state is ready.
    private var deepLinkUri by mutableStateOf<Uri?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        deepLinkUri = pickQuickpitikUri(intent)
        setContent {
            QuickPitikMobileTheme {
                val navController = rememberNavController()
                val authViewModel: AuthViewModel = viewModel()
                val cartViewModel: CartViewModel = viewModel()
                // Drive deep-link routing from Compose so we have access to
                // both the NavController and the cart sheet flags.
                LaunchedEffect(deepLinkUri) {
                    val uri = deepLinkUri ?: return@LaunchedEffect
                    handleQuickpitikUri(uri, navController, cartViewModel)
                    deepLinkUri = null
                }
                // Cold start with a cached JWT should land on the user's home
                // surface, not bounce them through login again. Same role→route
                // mapping as onLoginSuccess below, kept in one place. remember{}
                // so a later clearSession() can't re-key the NavHost mid-session.
                val sessionManager = remember { SessionManager.getInstance(this@MainActivity) }
                val startDestination = remember {
                    ViewMode.init(sessionManager)
                    when {
                        sessionManager.getAccessToken() == null -> "login"
                        // A photographer who killed the app while in runner
                        // view cold-starts back into runner view — starting in
                        // studio with the flag still set would strand the flag,
                        // because studio is legal for the true role and the
                        // guard would never fire.
                        isPhotographerRole(sessionManager.getUserRole()) &&
                            sessionManager.isRunnerView() -> "runner"
                        isPhotographerRole(sessionManager.getUserRole()) -> "studio"
                        else -> "runner"
                    }
                }
                val runnerView by ViewMode.runnerView.collectAsState()

                // Raised by TokenAuthenticator when refresh fails: the session is
                // unrecoverable, so drop the whole back stack and land on login.
                // The payload (nullable) says WHY — e.g. an ACCOUNT_SUSPENDED
                // rejection — and is surfaced as a notice on the login screen.
                var sessionNotice by remember { mutableStateOf<String?>(null) }
                LaunchedEffect(Unit) {
                    SessionEvents.forcedLogout.collect { reason ->
                        sessionNotice = reason
                        // Parity with the manual sign-out paths: the next user
                        // on this device must not inherit the previous
                        // session's cart pill — nor its runner-view flag.
                        cartViewModel.clearCart()
                        ViewMode.reset(sessionManager)
                        navController.navigate("login") {
                            popUpTo(0) { inclusive = true }
                            launchSingleTop = true
                        }
                    }
                }

                val navBackStackEntry by navController.currentBackStackEntryAsState()
                val currentRoute = navBackStackEntry?.destination?.route
                // Exact route PATTERNS, not startsWith: "orders/return/{orderId}"
                // is the modal PayMongo receipt and must NOT carry the
                // persistent nav (startsWith("orders") used to match it).
                val showRunnerBottomBar = currentRoute in RUNNER_NAV_ROUTES

                // While any studio/* destination is on the stack this resolves
                // the graph's back-stack entry — the owner of the ONE shared
                // PhotographerDashboardViewModel. Null for runner sessions, so
                // a runner never constructs the VM (and its init{} salvo).
                val studioEntry = remember(navBackStackEntry) {
                    runCatching { navController.getBackStackEntry("studio") }.getOrNull()
                }

                // The single studio tab-navigation path: bottom nav + Overview
                // quick links both come through here, so the per-tab
                // deliberate-refresh `when` exists exactly once (it was
                // duplicated twice in the old DashboardScreen). popUpTo keeps
                // the stack at [home, currentTab] — back from any tab lands
                // Home; back from Home exits. saveState/restoreState preserve
                // tab-internal scroll. NOTE: studio/capture is deliberately
                // absent from the refresh `when` — PublicEventPickerList
                // refetches itself on re-entry.
                val studioNavigate: (String) -> Unit = navigate@{ route ->
                    if (currentRoute != route) {
                        navController.navigate(route) {
                            popUpTo("studio/home") { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    }
                    val entry = runCatching {
                        navController.getBackStackEntry("studio")
                    }.getOrNull() ?: return@navigate
                    val vm = ViewModelProvider(entry)[PhotographerDashboardViewModel::class.java]
                    when (route) {
                        "studio/home" -> {
                            vm.fetchVerificationStatus()
                            vm.fetchEvents()
                            vm.fetchEarningsAndTransactions()
                            // REST hydration for the inbox — the WS push in the
                            // VM only refetches once the socket actually opens,
                            // and Overview derives the rejection banner from
                            // these messages.
                            vm.fetchMessages()
                        }
                        "studio/events" -> {
                            vm.fetchEvents()
                            // The tab merges covered + public events; both
                            // refreshed here (this replaces the screen's old
                            // mount-time LaunchedEffect).
                            vm.fetchPublicEvents()
                        }
                        "studio/earnings" -> vm.fetchEarningsAndTransactions()
                        "studio/settings" -> {
                            vm.fetchVerificationStatus()
                            vm.fetchSettings()
                        }
                    }
                }

                // Inbox socket held once for the whole studio session — a
                // per-tab lifecycle would reconnect (and refetch) on every tab
                // switch. See StudioInboxLifecycle.
                if (studioEntry != null) {
                    val studioVm: PhotographerDashboardViewModel = viewModel(studioEntry)
                    StudioInboxLifecycle(studioVm)
                }

                // Role guard — the one choke point every navigation source
                // (bottom nav, deep links, programmatic) passes through.
                // Runner routes reject a photographer; studio routes reject a
                // runner. `photographer/{handle}` and the auth routes are in
                // neither set (shared). Redirects never popUpTo: a hostile
                // deep link must not be able to pop the studio graph and kill
                // a live tether session.
                // Web use-effective-role parity: runner routes check the
                // EFFECTIVE role (a photographer in runner view passes),
                // studio routes check the TRUE role (a photographer in runner
                // view may still deep-return to studio).
                LaunchedEffect(navBackStackEntry, runnerView) {
                    val route = navBackStackEntry?.destination?.route ?: return@LaunchedEffect
                    if (sessionManager.getAccessToken() == null) return@LaunchedEffect
                    val photographer = isPhotographerRole(sessionManager.getUserRole())
                    val effectiveRunner = !photographer || runnerView
                    when {
                        route in RUNNER_ROUTES && !effectiveRunner ->
                            navController.navigate("studio") { launchSingleTop = true }
                        route.startsWith("studio") && !photographer ->
                            navController.navigate("runner") { launchSingleTop = true }
                    }
                }

                // RunnerTopBar's "Switch to photographer" lands here — the bar
                // has no NavController. Pop-inclusive so the stack never mixes
                // the two roles' surfaces.
                LaunchedEffect(Unit) {
                    ViewMode.switchToPhotographer.collect {
                        ViewMode.reset(sessionManager)
                        navController.navigate("studio") {
                            popUpTo("runner") { inclusive = true }
                            launchSingleTop = true
                        }
                    }
                }

                Scaffold(
                    containerColor = Bone,
                    bottomBar = {
                        when {
                            currentRoute in STUDIO_TAB_ROUTES && studioEntry != null -> {
                                val studioVm: PhotographerDashboardViewModel = viewModel(studioEntry)
                                val verificationState by studioVm.verificationState.collectAsState()
                                val showSettingsBadge = when (val s = verificationState) {
                                    is VerificationUiState.Success ->
                                        s.verification.status.lowercase() != "approved"
                                    else -> true
                                }
                                PhotographerFloatingBottomNav(
                                    currentRoute = currentRoute,
                                    showSettingsBadge = showSettingsBadge,
                                    onNavigate = studioNavigate,
                                )
                            }
                            showRunnerBottomBar -> RunnerFloatingBottomNav(
                                currentRoute = currentRoute,
                                onNavigate = { route ->
                                    if (currentRoute != route) {
                                        navController.navigate(route) {
                                            if (route == "events") {
                                                popUpTo("events") { inclusive = false }
                                            }
                                            launchSingleTop = true
                                        }
                                    }
                                }
                            )
                        }
                    }
                ) { innerPadding ->
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(innerPadding)
                    ) {
                        NavHost(
                            navController = navController,
                            startDestination = startDestination
                        ) {
                            composable("login") {
                                LoginScreen(
                                    viewModel = authViewModel,
                                    sessionNotice = sessionNotice,
                                    onNavigateToRegister = {
                                        navController.navigate("register")
                                    },
                                    onNavigateToForgotPassword = {
                                        navController.navigate("forgot-password")
                                    },
                                    onLoginSuccess = { isPhotographer ->
                                        sessionNotice = null
                                        val target = if (isPhotographer) "studio" else "runner"
                                        if (!isPhotographer) cartViewModel.fetchCart()
                                        navController.navigate(target) {
                                            popUpTo("login") { inclusive = true }
                                        }
                                    }
                                )
                            }
                            composable("forgot-password") {
                                ForgotPasswordScreen(
                                    viewModel = authViewModel,
                                    onNavigateToLogin = {
                                        navController.navigate("login") {
                                            popUpTo("login") { inclusive = true }
                                        }
                                    }
                                )
                            }
                            composable("register") {
                                RegisterScreen(
                                    viewModel = authViewModel,
                                    onNavigateToLogin = {
                                        navController.navigate("login")
                                    },
                                    onRegisterSuccess = { isPhotographer ->
                                        val target = if (isPhotographer) "studio" else "runner"
                                        if (!isPhotographer) cartViewModel.fetchCart()
                                        navController.navigate(target) {
                                            popUpTo("login") { inclusive = true }
                                        }
                                    }
                                )
                            }
                            // ── Photographer studio ─────────────────────────
                            // Nested graph so all seven routes share ONE
                            // PhotographerDashboardViewModel scoped to the
                            // graph's back-stack entry: lazy (a runner session
                            // never constructs it), and cleared exactly when
                            // logout pops the graph — the same teardown the old
                            // dashboard-entry scoping gave the tether loop.
                            // GUARD RULE: nothing may popUpTo past "studio"
                            // except logout/forced-logout, or a live tether VM
                            // dies mid-shoot.
                            navigation(startDestination = "studio/home", route = "studio") {
                                val studioLogout: () -> Unit = {
                                    authViewModel.logout()
                                    cartViewModel.clearCart()
                                    navController.navigate("login") {
                                        popUpTo("studio") { inclusive = true }
                                    }
                                }
                                val openProfilePreview: () -> Unit = {
                                    navController.navigate("studio/profile-preview")
                                }
                                // Enter runner view: pop the studio graph
                                // inclusively so the stack never mixes the two
                                // roles' surfaces (system back can't land a
                                // runner-view user on a studio screen). This
                                // clears the studio VM — same teardown as
                                // logout; StudioTabScaffold confirms first if
                                // a shutter watch is live.
                                val switchToRunnerView: () -> Unit = {
                                    ViewMode.set(sessionManager, true)
                                    navController.navigate("runner") {
                                        popUpTo("studio") { inclusive = true }
                                        launchSingleTop = true
                                    }
                                }
                                composable("studio/home") { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    StudioTabScaffold(
                                        viewModel = vm,
                                        onLogout = studioLogout,
                                        onPreviewProfile = openProfilePreview,
                                        onSwitchToRunner = switchToRunnerView,
                                    ) {
                                        PhotographerOverviewScreen(
                                            viewModel = vm,
                                            onNavigateToSettings = { studioNavigate("studio/settings") },
                                            onNavigateToTab = studioNavigate,
                                        )
                                    }
                                }
                                composable("studio/capture") { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    StudioTabScaffold(
                                        viewModel = vm,
                                        onLogout = studioLogout,
                                        onPreviewProfile = openProfilePreview,
                                        onSwitchToRunner = switchToRunnerView,
                                    ) {
                                        PhotographerCaptureScreen(viewModel = vm)
                                    }
                                }
                                composable("studio/events") { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    StudioTabScaffold(
                                        viewModel = vm,
                                        onLogout = studioLogout,
                                        onPreviewProfile = openProfilePreview,
                                        onSwitchToRunner = switchToRunnerView,
                                    ) {
                                        PhotographerEventsScreen(
                                            viewModel = vm,
                                            onOpenShare = { event ->
                                                navController.navigate("studio/share/${event.id}")
                                            },
                                            onSyncEvent = { ev ->
                                                vm.selectEvent(ev)
                                                studioNavigate("studio/capture")
                                            },
                                        )
                                    }
                                }
                                composable("studio/earnings") { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    StudioTabScaffold(
                                        viewModel = vm,
                                        onLogout = studioLogout,
                                        onPreviewProfile = openProfilePreview,
                                        onSwitchToRunner = switchToRunnerView,
                                    ) {
                                        PhotographerEarningsScreen(viewModel = vm)
                                    }
                                }
                                composable("studio/settings") { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    StudioTabScaffold(
                                        viewModel = vm,
                                        onLogout = studioLogout,
                                        onPreviewProfile = openProfilePreview,
                                        onSwitchToRunner = switchToRunnerView,
                                    ) {
                                        PhotographerSettingsScreen(
                                            viewModel = vm,
                                            onLogout = studioLogout,
                                        )
                                    }
                                }
                                // Fullscreen sub-surfaces — no tab chrome, no
                                // bottom nav; back pops to the launching tab.
                                composable(
                                    route = "studio/share/{eventId}",
                                    arguments = listOf(navArgument("eventId") { type = NavType.StringType }),
                                ) { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    val eventId = entry.arguments?.getString("eventId")
                                    val eventsState by vm.eventsState.collectAsState()
                                    val event = (eventsState as? EventsState.Success)
                                        ?.events?.firstOrNull { it.id == eventId }
                                    if (event == null) {
                                        // List not loaded / unknown id — nothing
                                        // to share; fall back to the tab.
                                        LaunchedEffect(eventId) { navController.popBackStack() }
                                    } else {
                                        StudioTheme {
                                            PhotographerEventShareScreen(
                                                event = event,
                                                viewModel = vm,
                                                onBack = { navController.popBackStack() },
                                            )
                                        }
                                    }
                                }
                                composable("studio/profile-preview") { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    // Route-scoped on purpose: the same screen
                                    // serves a runner tapping a photo byline
                                    // via "photographer/{handle}".
                                    val publicVm: PublicPhotographerViewModel = viewModel()
                                    val brandSettings by vm.brandSettings.collectAsState()
                                    LaunchedEffect(Unit) {
                                        if (brandSettings == null) vm.fetchBrandSettings()
                                    }
                                    StudioTheme {
                                        PhotographerPublicProfileScreen(
                                            handle = brandSettings?.handle,
                                            viewModel = publicVm,
                                            onBack = { navController.popBackStack() },
                                        )
                                    }
                                }
                            }
                            // Runner state belongs to the authenticated runner
                            // session, not the Activity. The graph is lazy and
                            // popping it on logout clears every user-owned VM.
                            navigation(startDestination = "events", route = "runner") {
                                val runnerLogout: () -> Unit = {
                                    authViewModel.logout()
                                    cartViewModel.clearCart()
                                    navController.navigate("login") {
                                        popUpTo("runner") { inclusive = true }
                                    }
                                }
                                composable("events") { entry ->
                                    val graphEntry = runnerGraphEntry(navController, entry)
                                    val runnerViewModel: RunnerGalleryViewModel = viewModel(graphEntry)
                                    val savedEventsViewModel: SavedEventsViewModel = viewModel(graphEntry)
                                    val runnerInboxViewModel: RunnerInboxViewModel = viewModel(graphEntry)
                                EventsDiscoveryScreen(
                                    viewModel = runnerViewModel,
                                    savedEventsViewModel = savedEventsViewModel,
                                    inboxViewModel = runnerInboxViewModel,
                                    onEventSelected = { event ->
                                        runnerViewModel.selectEvent(event)
                                        navController.navigate("gallery")
                                    },
                                    onOpenOrder = { orderId -> navController.navigate("orders?orderId=$orderId") },
                                    onLogout = runnerLogout,
                                )
                            }
                                composable("gallery") { entry ->
                                    val graphEntry = runnerGraphEntry(navController, entry)
                                    val runnerViewModel: RunnerGalleryViewModel = viewModel(graphEntry)
                                    val savedEventsViewModel: SavedEventsViewModel = viewModel(graphEntry)
                                    val runnerInboxViewModel: RunnerInboxViewModel = viewModel(graphEntry)
                                RunnerGalleryScreen(
                                    viewModel = runnerViewModel,
                                    cartViewModel = cartViewModel,
                                    inboxViewModel = runnerInboxViewModel,
                                    savedEventsViewModel = savedEventsViewModel,
                                    onOpenOrder = { orderId ->
                                        navController.navigate("orders?orderId=$orderId")
                                    },
                                    onOpenPhotographer = { handle ->
                                        navController.navigate("photographer/$handle")
                                    },
                                    onNavigateToProfile = {
                                        navController.navigate("profile")
                                    },
                                    onNavigateBack = {
                                        navController.popBackStack()
                                    },
                                    onLogout = runnerLogout,
                                )
                            }
                            composable(
                                route = "photographer/{handle}",
                                arguments = listOf(navArgument("handle") { type = NavType.StringType }),
                            ) { entry ->
                                val publicPhotographerViewModel: PublicPhotographerViewModel = viewModel()
                                PhotographerPublicProfileScreen(
                                    handle = entry.arguments?.getString("handle"),
                                    viewModel = publicPhotographerViewModel,
                                    onBack = { navController.popBackStack() },
                                    // Runner context — the per-event gallery is
                                    // transactional here (web parity).
                                    cartViewModel = cartViewModel,
                                )
                            }
                                composable("profile") { entry ->
                                    val graphEntry = runnerGraphEntry(navController, entry)
                                    val profileViewModel: ProfileViewModel = viewModel(graphEntry)
                                    val runnerViewModel: RunnerGalleryViewModel = viewModel(graphEntry)
                                    val savedEventsViewModel: SavedEventsViewModel = viewModel(graphEntry)
                                ProfileScreen(
                                    viewModel = profileViewModel,
                                    cartViewModel = cartViewModel,
                                    savedEventsViewModel = savedEventsViewModel,
                                    onOpenEvent = { slug ->
                                        val event = runnerViewModel.eventBySlug(slug)
                                        if (event != null) {
                                            runnerViewModel.selectEvent(event)
                                            navController.navigate("gallery")
                                        } else {
                                            navController.navigate("events")
                                        }
                                    },
                                    onBrowseEvents = {
                                        navController.navigate("events")
                                    },
                                    onLogout = runnerLogout,
                                )
                            }
                                composable("settings") { entry ->
                                    val graphEntry = runnerGraphEntry(navController, entry)
                                    val profileViewModel: ProfileViewModel = viewModel(graphEntry)
                                AccountSettingsScreen(
                                    viewModel = profileViewModel,
                                    onLogout = runnerLogout,
                                )
                            }
                            composable(
                                route = "orders?orderId={orderId}",
                                arguments = listOf(
                                    navArgument("orderId") {
                                        type = NavType.StringType
                                        nullable = true
                                        defaultValue = null
                                    }
                                ),
                            ) { entry ->
                                OrdersScreen(
                                    viewModel = cartViewModel,
                                    initialOrderId = entry.arguments?.getString("orderId"),
                                    onLogout = runnerLogout,
                                )
                            }
                            composable(
                                route = "orders/return/{orderId}?token={token}",
                                arguments = listOf(
                                    navArgument("orderId") { type = NavType.StringType },
                                    navArgument("token") {
                                        type = NavType.StringType
                                        nullable = true
                                        defaultValue = null
                                    }
                                ),
                            ) { entry ->
                                val orderId = entry.arguments?.getString("orderId").orEmpty()
                                val shareToken = entry.arguments?.getString("token")
                                OrderReturnScreen(
                                    orderId = orderId,
                                    shareToken = shareToken,
                                    cartViewModel = cartViewModel,
                                    onNavigateToOrders = {
                                        cartViewModel.resetOrderReturnState()
                                        navController.navigate("orders") {
                                            popUpTo("events") { saveState = true }
                                            launchSingleTop = true
                                        }
                                    },
                                    onBrowseEvents = {
                                        cartViewModel.resetOrderReturnState()
                                        navController.navigate("events") {
                                            popUpTo("events") { inclusive = true }
                                            launchSingleTop = true
                                        }
                                    },
                                    onClose = {
                                        cartViewModel.resetOrderReturnState()
                                        navController.popBackStack()
                                    },
                                )
                            }

                            }
                        }

                        // Global floating-cart pill + cart/checkout sheets
                        FloatingCart(
                            navController = navController,
                            cartViewModel = cartViewModel,
                            onAfterCheckoutSuccess = {
                                navController.navigate("orders") {
                                    popUpTo("events") { saveState = true }
                                    launchSingleTop = true
                                }
                            },
                        )
                    }
                }
            }
        }
    }

    // singleTask launchMode means new deep-link intents land here instead of
    // spawning a new MainActivity — capture the URI and let Compose route.
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        pickQuickpitikUri(intent)?.let { deepLinkUri = it }
    }

    private fun pickQuickpitikUri(intent: Intent?): Uri? {
        val data = intent?.data ?: return null
        return if (data.scheme.equals("quickpitik", ignoreCase = true)) data else null
    }

    // Maps:
    //   quickpitik://orders/return?orderId=…&token=…  →  orders/return/{orderId}
    //                                                    (the receipt screen)
    //   quickpitik://cart?orderId=…                   →  re-open cart sheet
    private fun handleQuickpitikUri(
        uri: Uri,
        navController: NavController,
        cartViewModel: CartViewModel,
    ) {
        // Host carries the route — Android parses "quickpitik://orders/return"
        // as scheme=quickpitik, host=orders, path=/return.
        val host = uri.host?.lowercase() ?: return
        val path = uri.path?.lowercase().orEmpty()
        // Every quickpitik:// target below is a runner surface. A photographer
        // token would 403 the cart fetch and land on screens with no way back,
        // so route them home instead. Handled here (not only in the route
        // guard) because the `cart` case opens a sheet WITHOUT navigating —
        // the destination-change guard never sees it. launchSingleTop, never
        // popUpTo: a stray deep link must not pop a live tether session.
        val session = SessionManager.getInstance(this)
        if (isPhotographerRole(session.getUserRole()) && !session.isRunnerView()) {
            navController.navigate("studio") { launchSingleTop = true }
            return
        }
        when {
            host == "orders" && path.startsWith("/return") -> {
                val orderId = uri.getQueryParameter("orderId")
                cartViewModel.closeCartSheet()
                cartViewModel.closeCheckoutSheet()
                cartViewModel.resetCheckoutState()
                if (orderId.isNullOrBlank()) {
                    // Deep link malformed — fall back to the orders list so the
                    // user can find their purchase manually instead of dead-ending.
                    cartViewModel.fetchCart()
                    navController.navigate("orders") {
                        popUpTo("events") { saveState = true }
                        launchSingleTop = true
                    }
                } else {
                    val token = uri.getQueryParameter("token")
                    val route = if (token.isNullOrBlank()) "orders/return/$orderId" else "orders/return/$orderId?token=$token"
                    navController.navigate(route) {
                        popUpTo("events") { saveState = true }
                        launchSingleTop = true
                    }
                }

            }
            host == "orders" -> {
                cartViewModel.closeCartSheet()
                cartViewModel.closeCheckoutSheet()
                cartViewModel.resetCheckoutState()
                cartViewModel.fetchCart()
                navController.navigate("orders") {
                    popUpTo("events") { saveState = true }
                    launchSingleTop = true
                }
            }
            host == "cart" -> {
                cartViewModel.closeCheckoutSheet()
                cartViewModel.resetCheckoutState()
                cartViewModel.fetchCart()
                cartViewModel.openCartSheet()
            }
        }
    }
}

// The ONE shared studio VM, owned by the "studio" graph's back-stack entry —
// every studio route resolves the same instance, and it is cleared (tether
// teardown included) exactly when logout pops the graph. `remember(entry)`
// because getBackStackEntry must not run on every recomposition.
@Composable
private fun studioViewModel(
    navController: NavHostController,
    entry: NavBackStackEntry,
): PhotographerDashboardViewModel {
    val graphEntry = remember(entry) { navController.getBackStackEntry("studio") }
    return viewModel(graphEntry)
}

@Composable
private fun runnerGraphEntry(
    navController: NavHostController,
    entry: NavBackStackEntry,
): NavBackStackEntry = remember(entry) { navController.getBackStackEntry("runner") }

// ─── Floating-pill bottom nav for Runner ──────────────────────────────────────
// Mirrors the Quiet Studio photographer floating pill nav format:
// Bone background, 1dp Line border, PillShape, animated Ink active background, Bone/Slate tint.
@Composable
private fun RunnerFloatingBottomNav(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
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
            RunnerFloatingNavItem(
                icon = Icons.Default.Search,
                label = "Browse",
                selected = currentRoute == "events" || currentRoute == "gallery",
                onClick = { onNavigate("events") },
                modifier = Modifier.weight(1f),
            )
            RunnerFloatingNavItem(
                icon = Icons.Default.Face,
                label = "Profile",
                selected = currentRoute == "profile",
                onClick = { onNavigate("profile") },
                modifier = Modifier.weight(1f),
            )
            RunnerFloatingNavItem(
                icon = Icons.Default.List,
                label = "Orders",
                // Exact pattern — startsWith("orders") also matched the
                // PayMongo receipt route "orders/return/{orderId}".
                selected = currentRoute == "orders?orderId={orderId}",
                onClick = { onNavigate("orders") },
                modifier = Modifier.weight(1f),
            )
            RunnerFloatingNavItem(
                icon = Icons.Default.Settings,
                label = "Settings",
                selected = currentRoute == "settings",
                onClick = { onNavigate("settings") },
                modifier = Modifier.weight(1f),
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun RunnerFloatingNavItem(
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
        label = "runnerNavItemBg",
    )
    val tint by animateColorAsState(
        targetValue = if (selected) Bone else Slate,
        animationSpec = tween(180),
        label = "runnerNavItemTint",
    )
    Column(
        modifier = modifier
            .heightIn(min = 56.dp)
            .clip(PillShape)
            .background(bg)
            .clickable(onClick = onClick)
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
