package com.quickpitik.mobile

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.FastOutLinearInEasing
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
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
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import android.app.Application
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavBackStackEntry
import androidx.navigation.NavController
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
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
import com.quickpitik.mobile.ui.auth.ConfirmEmailChangeScreen
import com.quickpitik.mobile.ui.auth.VerifyEmailScreen
import com.quickpitik.mobile.ui.photographer.EventsState
import com.quickpitik.mobile.ui.photographer.PhotographerCaptureScreen
import com.quickpitik.mobile.ui.photographer.PhotographerDashboardViewModel
import com.quickpitik.mobile.ui.photographer.PhotographerEarningsScreen
import com.quickpitik.mobile.ui.photographer.PhotographerEventFormScreen
import com.quickpitik.mobile.ui.photographer.PhotographerEventShareScreen
import com.quickpitik.mobile.ui.photographer.PhotographerEventsScreen
import com.quickpitik.mobile.ui.photographer.PhotographerFloatingBottomNav
import com.quickpitik.mobile.ui.photographer.PhotographerOverviewScreen
import com.quickpitik.mobile.ui.photographer.PhotographerPublicProfileScreen
import com.quickpitik.mobile.ui.photographer.PhotographerSettingsScreen
import com.quickpitik.mobile.ui.photographer.PublicPhotographerViewModel
import com.quickpitik.mobile.ui.photographer.STUDIO_TAB_ROUTES
import com.quickpitik.mobile.ui.photographer.StudioEarningsViewModel
import com.quickpitik.mobile.ui.photographer.StudioInboxLifecycle
import com.quickpitik.mobile.ui.photographer.StudioSettingsViewModel
import com.quickpitik.mobile.ui.photographer.StudioTabScaffold
import com.quickpitik.mobile.ui.photographer.StudioTheme
import com.quickpitik.mobile.ui.photographer.TetherViewModel
import com.quickpitik.mobile.ui.photographer.VerificationUiState
import com.quickpitik.mobile.ui.runner.AccountSettingsScreen
import com.quickpitik.mobile.ui.runner.CartViewModel
import com.quickpitik.mobile.ui.runner.EventsDiscoveryScreen
import com.quickpitik.mobile.ui.runner.FloatingCart
import com.quickpitik.mobile.ui.runner.OrderReturnScreen
import com.quickpitik.mobile.ui.runner.OrdersScreen
import com.quickpitik.mobile.ui.runner.ProfileScreen
import com.quickpitik.mobile.ui.runner.ProfileViewModel
import com.quickpitik.mobile.ui.runner.RunnerGalleryScreen
import com.quickpitik.mobile.ui.runner.RunnerGalleryViewModel
import com.quickpitik.mobile.ui.runner.RunnerInboxViewModel
import com.quickpitik.mobile.ui.runner.SavedEventsViewModel
import com.quickpitik.mobile.ui.runner.VerifyScreen
import com.quickpitik.mobile.ui.runner.VerifyViewModel
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.QuickPitikMobileTheme
import com.quickpitik.mobile.ui.theme.Slate

// Runner-owned route PATTERNS (as the back stack reports them). Used by the
// role guard and, minus the receipt route, by the bottom-nav gate.
// "photographer/{handle}" and the auth routes belong to NEITHER role set —
// they are shared surfaces.
// The receipt route carries the optional guest ?token=. One constant feeds both
// the NavHost registration and this guard set so the two can never drift (F5:
// the bare "orders/return/{orderId}" had once dropped out of the guard).
internal const val RECEIPT_ROUTE_PATTERN = "orders/return/{orderId}?token={token}"

internal val RUNNER_ROUTES = setOf(
    "events", "gallery", "profile", "settings", "verify",
    "orders?orderId={orderId}", RECEIPT_ROUTE_PATTERN,
)

// Deep-link ids must be path/query-safe before being spliced into a route
// string — the scheme-only filter is exported, so a hostile quickpitik:// could
// otherwise inject a "/" or space that breaks NavController route parsing (F6).
private val DEEP_LINK_ID = Regex("^[A-Za-z0-9._-]{1,64}$")

// Where the runner bottom nav shows: the tab surfaces + the orders list, but
// NOT the modal PayMongo receipt ("orders/return/{orderId}").
internal val RUNNER_NAV_ROUTES = setOf(
    "events", "gallery", "profile", "settings", "orders?orderId={orderId}",
)

// Tab route sequences in left-to-right display order for directional slide transitions
private val RUNNER_TAB_ORDER = listOf(
    "events",
    "profile",
    "orders?orderId={orderId}",
    "settings",
)

private val STUDIO_TAB_ORDER = listOf(
    "studio/home",
    "studio/capture",
    "studio/events",
    "studio/earnings",
    "studio/settings",
)

internal enum class BottomBar { None, Studio, Runner }

/**
 * Which floating nav a route carries. Studio tabs need the studio graph entry
 * (the owner of the shared dashboard ViewModel) to exist; the runner set is
 * exact route PATTERNS, never startsWith — "orders/return/{orderId}" is the
 * modal PayMongo receipt and must stay bare (startsWith("orders") matched it
 * once). Everything else — login, the receipt, a public profile — gets none.
 */
internal fun bottomBarFor(currentRoute: String?, hasStudioEntry: Boolean): BottomBar = when {
    currentRoute in STUDIO_TAB_ROUTES && hasStudioEntry -> BottomBar.Studio
    currentRoute in RUNNER_NAV_ROUTES -> BottomBar.Runner
    else -> BottomBar.None
}

internal fun tabIndexForRoute(route: String?): Int {
    if (route == null) return -1
    val clean = route.substringBefore("?")
    val rIdx = RUNNER_TAB_ORDER.indexOfFirst { it.substringBefore("?") == clean }
    if (rIdx != -1) return rIdx
    val sIdx = STUDIO_TAB_ORDER.indexOfFirst { it.substringBefore("?") == clean }
    if (sIdx != -1) return sIdx
    return -1
}

internal fun arePeerTabs(fromRoute: String?, toRoute: String?): Boolean {
    if (fromRoute == null || toRoute == null) return false
    val f = fromRoute.substringBefore("?")
    val t = toRoute.substringBefore("?")
    val bothRunner = RUNNER_TAB_ORDER.any { it.substringBefore("?") == f } &&
        RUNNER_TAB_ORDER.any { it.substringBefore("?") == t }
    if (bothRunner) return true
    val bothStudio = STUDIO_TAB_ORDER.any { it.substringBefore("?") == f } &&
        STUDIO_TAB_ORDER.any { it.substringBefore("?") == t }
    return bothStudio
}

class MainActivity : ComponentActivity() {
    // Latest deep-link URI from a quickpitik:// intent. Compose observes this
    // via a LaunchedEffect and routes the user to OrdersScreen (return) or
    // back to the cart sheet (cancel) once the auth state is ready.
    private var deepLinkUri by mutableStateOf<Uri?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Only consume the launch intent's deep link on a FRESH start. On a
        // config change / process-death restore the NavController restores its
        // own back stack — re-handling the intent here would re-navigate and,
        // for one-shot email tokens, re-POST an already-spent token.
        if (savedInstanceState == null) {
            deepLinkUri = pickQuickpitikUri(intent)
        }
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
                    startDestinationFor(
                        signedIn = sessionManager.getAccessToken() != null,
                        photographer = isPhotographerRole(sessionManager.getUserRole()),
                        runnerView = sessionManager.isRunnerView(),
                    )
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
                    val owner = ViewModelProvider(entry)
                    when (route) {
                        "studio/home" -> {
                            vm.fetchVerificationStatus()
                            vm.fetchEvents()
                            owner[StudioEarningsViewModel::class.java].fetchEarningsAndTransactions()
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
                        "studio/earnings" ->
                            owner[StudioEarningsViewModel::class.java].fetchEarningsAndTransactions()
                        "studio/settings" -> {
                            vm.fetchVerificationStatus()
                            owner[StudioSettingsViewModel::class.java].fetchSettings()
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
                    guardRedirect(isPhotographerRole(sessionManager.getUserRole()), runnerView, route)
                        ?.let { navController.navigate(it) { launchSingleTop = true } }
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
                        when (bottomBarFor(currentRoute, studioEntry != null)) {
                            BottomBar.Studio -> studioEntry?.let { entry ->
                                val studioVm: PhotographerDashboardViewModel = viewModel(entry)
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
                            BottomBar.Runner -> RunnerFloatingBottomNav(
                                currentRoute = currentRoute,
                                onNavigate = { route ->
                                    if (route == "events" && currentRoute == "gallery") {
                                        if (!navController.popBackStack("events", inclusive = false)) {
                                            navController.navigate("events") {
                                                popUpTo("events") { inclusive = true }
                                                launchSingleTop = true
                                            }
                                        }
                                        return@RunnerFloatingBottomNav
                                    }
                                    val isAlreadyOnOrders = route == "orders" && currentRoute?.startsWith("orders") == true
                                    if (currentRoute != route && !isAlreadyOnOrders) {
                                        navController.navigate(route) {
                                            popUpTo("events") { saveState = true }
                                            launchSingleTop = true
                                            restoreState = true
                                        }
                                    }
                                }
                            )
                            BottomBar.None -> Unit
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
                            startDestination = startDestination,
                            enterTransition = {
                                val fromRoute = initialState.destination.route
                                val toRoute = targetState.destination.route
                                if (arePeerTabs(fromRoute, toRoute)) {
                                    val fromIdx = tabIndexForRoute(fromRoute)
                                    val toIdx = tabIndexForRoute(toRoute)
                                    if (toIdx > fromIdx) {
                                        slideIntoContainer(
                                            towards = AnimatedContentTransitionScope.SlideDirection.Start,
                                            animationSpec = tween(280, easing = FastOutSlowInEasing),
                                            initialOffset = { (it * 0.28f).toInt() }
                                        ) + fadeIn(animationSpec = tween(240, easing = FastOutSlowInEasing))
                                    } else {
                                        slideIntoContainer(
                                            towards = AnimatedContentTransitionScope.SlideDirection.End,
                                            animationSpec = tween(280, easing = FastOutSlowInEasing),
                                            initialOffset = { (it * 0.28f).toInt() }
                                        ) + fadeIn(animationSpec = tween(240, easing = FastOutSlowInEasing))
                                    }
                                } else {
                                    slideIntoContainer(
                                        towards = AnimatedContentTransitionScope.SlideDirection.Start,
                                        animationSpec = tween(300, easing = FastOutSlowInEasing)
                                    ) + fadeIn(animationSpec = tween(240, easing = FastOutSlowInEasing))
                                }
                            },
                            exitTransition = {
                                val fromRoute = initialState.destination.route
                                val toRoute = targetState.destination.route
                                if (arePeerTabs(fromRoute, toRoute)) {
                                    val fromIdx = tabIndexForRoute(fromRoute)
                                    val toIdx = tabIndexForRoute(toRoute)
                                    if (toIdx > fromIdx) {
                                        slideOutOfContainer(
                                            towards = AnimatedContentTransitionScope.SlideDirection.Start,
                                            animationSpec = tween(240, easing = FastOutLinearInEasing),
                                            targetOffset = { (it * 0.28f).toInt() }
                                        ) + fadeOut(animationSpec = tween(200, easing = FastOutLinearInEasing))
                                    } else {
                                        slideOutOfContainer(
                                            towards = AnimatedContentTransitionScope.SlideDirection.End,
                                            animationSpec = tween(240, easing = FastOutLinearInEasing),
                                            targetOffset = { (it * 0.28f).toInt() }
                                        ) + fadeOut(animationSpec = tween(200, easing = FastOutLinearInEasing))
                                    }
                                } else {
                                    slideOutOfContainer(
                                        towards = AnimatedContentTransitionScope.SlideDirection.Start,
                                        animationSpec = tween(300, easing = FastOutSlowInEasing),
                                        targetOffset = { (it * 0.25f).toInt() }
                                    ) + fadeOut(animationSpec = tween(200, easing = FastOutLinearInEasing))
                                }
                            },
                            popEnterTransition = {
                                val fromRoute = initialState.destination.route
                                val toRoute = targetState.destination.route
                                if (arePeerTabs(fromRoute, toRoute)) {
                                    val fromIdx = tabIndexForRoute(fromRoute)
                                    val toIdx = tabIndexForRoute(toRoute)
                                    if (toIdx > fromIdx) {
                                        slideIntoContainer(
                                            towards = AnimatedContentTransitionScope.SlideDirection.Start,
                                            animationSpec = tween(280, easing = FastOutSlowInEasing),
                                            initialOffset = { (it * 0.28f).toInt() }
                                        ) + fadeIn(animationSpec = tween(240, easing = FastOutSlowInEasing))
                                    } else {
                                        slideIntoContainer(
                                            towards = AnimatedContentTransitionScope.SlideDirection.End,
                                            animationSpec = tween(280, easing = FastOutSlowInEasing),
                                            initialOffset = { (it * 0.28f).toInt() }
                                        ) + fadeIn(animationSpec = tween(240, easing = FastOutSlowInEasing))
                                    }
                                } else {
                                    slideIntoContainer(
                                        towards = AnimatedContentTransitionScope.SlideDirection.End,
                                        animationSpec = tween(300, easing = FastOutSlowInEasing),
                                        initialOffset = { (it * 0.25f).toInt() }
                                    ) + fadeIn(animationSpec = tween(240, easing = FastOutSlowInEasing))
                                }
                            },
                            popExitTransition = {
                                val fromRoute = initialState.destination.route
                                val toRoute = targetState.destination.route
                                if (arePeerTabs(fromRoute, toRoute)) {
                                    val fromIdx = tabIndexForRoute(fromRoute)
                                    val toIdx = tabIndexForRoute(toRoute)
                                    if (toIdx > fromIdx) {
                                        slideOutOfContainer(
                                            towards = AnimatedContentTransitionScope.SlideDirection.Start,
                                            animationSpec = tween(240, easing = FastOutLinearInEasing),
                                            targetOffset = { (it * 0.28f).toInt() }
                                        ) + fadeOut(animationSpec = tween(200, easing = FastOutLinearInEasing))
                                    } else {
                                        slideOutOfContainer(
                                            towards = AnimatedContentTransitionScope.SlideDirection.End,
                                            animationSpec = tween(240, easing = FastOutLinearInEasing),
                                            targetOffset = { (it * 0.28f).toInt() }
                                        ) + fadeOut(animationSpec = tween(200, easing = FastOutLinearInEasing))
                                    }
                                } else {
                                    slideOutOfContainer(
                                        towards = AnimatedContentTransitionScope.SlideDirection.End,
                                        animationSpec = tween(300, easing = FastOutSlowInEasing)
                                    ) + fadeOut(animationSpec = tween(200, easing = FastOutLinearInEasing))
                                }
                            }
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
                            composable(
                                route = "verify-email?token={token}",
                                arguments = listOf(navArgument("token") { type = NavType.StringType; nullable = true; defaultValue = null })
                            ) { backStackEntry ->
                                val token = backStackEntry.arguments?.getString("token")
                                VerifyEmailScreen(
                                    // Its own AuthViewModel, scoped to this entry
                                    // (not the shared authViewModel): one send per
                                    // visit, the answer kept in the entry's saved
                                    // state; a later tap of the link is a new visit.
                                    viewModel = viewModel(backStackEntry),
                                    token = token,
                                    // popUpTo(0): the deep link can arrive with
                                    // any stack underneath (login OR a live
                                    // runner/studio graph) — popping to a named
                                    // route would silently no-op when it isn't
                                    // on the stack and leave dead screens behind.
                                    onNavigateToLogin = {
                                        navController.navigate("login") {
                                            popUpTo(0) { inclusive = true }
                                            launchSingleTop = true
                                        }
                                    },
                                    onNavigateToDashboard = { isPhotographer ->
                                        sessionNotice = null
                                        val target = if (isPhotographer) "studio" else "runner"
                                        if (!isPhotographer) cartViewModel.fetchCart()
                                        navController.navigate(target) {
                                            popUpTo(0) { inclusive = true }
                                            launchSingleTop = true
                                        }
                                    }
                                )
                            }
                            composable(
                                route = "confirm-email-change?token={token}",
                                arguments = listOf(navArgument("token") { type = NavType.StringType; nullable = true; defaultValue = null })
                            ) { backStackEntry ->
                                val token = backStackEntry.arguments?.getString("token")
                                ConfirmEmailChangeScreen(
                                    // Entry-scoped, same as verify-email above.
                                    viewModel = viewModel(backStackEntry),
                                    token = token,
                                    onNavigateToLogin = {
                                        // On success confirmEmailChange cleared the
                                        // session — finish the logout the same
                                        // way forcedLogout does: cart + the
                                        // in-memory runner-view flag, which
                                        // survives the prefs wipe.
                                        if (sessionManager.getAccessToken() == null) {
                                            cartViewModel.clearCart()
                                            ViewMode.reset(sessionManager)
                                        }
                                        navController.navigate("login") {
                                            popUpTo(0) { inclusive = true }
                                            launchSingleTop = true
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
                                    val settingsVm = studioSettingsViewModel(navController, entry)
                                    StudioTabScaffold(
                                        viewModel = vm,
                                        settingsViewModel = settingsVm,
                                        tetherViewModel = studioTetherViewModel(navController, entry),
                                        onLogout = studioLogout,
                                        onPreviewProfile = openProfilePreview,
                                        onSwitchToRunner = switchToRunnerView,
                                    ) {
                                        PhotographerOverviewScreen(
                                            viewModel = vm,
                                            settingsViewModel = settingsVm,
                                            earningsViewModel = studioEarningsViewModel(navController, entry),
                                            onNavigateToSettings = { studioNavigate("studio/settings") },
                                            onNavigateToTab = studioNavigate,
                                        )
                                    }
                                }
                                composable("studio/capture") { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    val tetherVm = studioTetherViewModel(navController, entry)
                                    StudioTabScaffold(
                                        viewModel = vm,
                                        settingsViewModel = studioSettingsViewModel(navController, entry),
                                        tetherViewModel = tetherVm,
                                        onLogout = studioLogout,
                                        onPreviewProfile = openProfilePreview,
                                        onSwitchToRunner = switchToRunnerView,
                                    ) {
                                        PhotographerCaptureScreen(
                                            viewModel = vm,
                                            tetherViewModel = tetherVm,
                                        )
                                    }
                                }
                                composable("studio/events") { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    val tetherVm = studioTetherViewModel(navController, entry)
                                    StudioTabScaffold(
                                        viewModel = vm,
                                        settingsViewModel = studioSettingsViewModel(navController, entry),
                                        tetherViewModel = tetherVm,
                                        onLogout = studioLogout,
                                        onPreviewProfile = openProfilePreview,
                                        onSwitchToRunner = switchToRunnerView,
                                    ) {
                                        PhotographerEventsScreen(
                                            viewModel = vm,
                                            tetherViewModel = tetherVm,
                                            onOpenShare = { event ->
                                                navController.navigate("studio/share/${event.id}")
                                            },
                                            onSyncEvent = { ev ->
                                                tetherVm.selectEvent(ev)
                                                studioNavigate("studio/capture")
                                            },
                                            onCreateEvent = {
                                                navController.navigate("studio/event-form")
                                            },
                                            onEditEvent = { ev ->
                                                navController.navigate("studio/event-form?eventId=${ev.id}")
                                            },
                                        )
                                    }
                                }
                                composable("studio/earnings") { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    StudioTabScaffold(
                                        viewModel = vm,
                                        settingsViewModel = studioSettingsViewModel(navController, entry),
                                        tetherViewModel = studioTetherViewModel(navController, entry),
                                        onLogout = studioLogout,
                                        onPreviewProfile = openProfilePreview,
                                        onSwitchToRunner = switchToRunnerView,
                                    ) {
                                        PhotographerEarningsScreen(
                                            viewModel = vm,
                                            earningsViewModel = studioEarningsViewModel(navController, entry),
                                        )
                                    }
                                }
                                composable("studio/settings") { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    val settingsVm = studioSettingsViewModel(navController, entry)
                                    StudioTabScaffold(
                                        viewModel = vm,
                                        settingsViewModel = settingsVm,
                                        tetherViewModel = studioTetherViewModel(navController, entry),
                                        onLogout = studioLogout,
                                        onPreviewProfile = openProfilePreview,
                                        onSwitchToRunner = switchToRunnerView,
                                    ) {
                                        PhotographerSettingsScreen(
                                            viewModel = vm,
                                            settingsViewModel = settingsVm,
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
                                composable(
                                    route = "studio/event-form?eventId={eventId}",
                                    arguments = listOf(navArgument("eventId") {
                                        type = NavType.StringType
                                        nullable = true
                                        defaultValue = null
                                    }),
                                ) { entry ->
                                    val vm = studioViewModel(navController, entry)
                                    val eventId = entry.arguments?.getString("eventId")
                                    StudioTheme {
                                        PhotographerEventFormScreen(
                                            viewModel = vm,
                                            eventId = eventId,
                                            onBack = { navController.popBackStack() },
                                            onSaved = { navController.popBackStack() },
                                        )
                                    }
                                }
                                composable("studio/profile-preview") { entry ->
                                    val settingsVm = studioSettingsViewModel(navController, entry)
                                    // Route-scoped on purpose: the same screen
                                    // serves a runner tapping a photo byline
                                    // via "photographer/{handle}".
                                    val publicVm: PublicPhotographerViewModel = viewModel()
                                    val brandSettings by settingsVm.brandSettings.collectAsState()
                                    val isBrandSettingsLoading by settingsVm.isFetchingBrandSettings.collectAsState()
                                    val brandSettingsError by settingsVm.settingsLoadError.collectAsState()
                                    LaunchedEffect(Unit) {
                                        if (brandSettings == null) settingsVm.fetchBrandSettings()
                                    }
                                    StudioTheme {
                                        PhotographerPublicProfileScreen(
                                            handle = brandSettings?.handle,
                                            viewModel = publicVm,
                                            onBack = { navController.popBackStack() },
                                            // "|| brandSettings == null": the fetch
                                            // starts in a LaunchedEffect AFTER the
                                            // first frame, so on that frame the
                                            // loading flag is still false — without
                                            // this the no-handle card flashes once.
                                            isBrandSettingsLoading = isBrandSettingsLoading ||
                                                (brandSettings == null && brandSettingsError == null),
                                            brandSettingsError = brandSettingsError,
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
                                val graphEntry = runnerGraphEntry(navController, entry)
                                val runnerViewModel: RunnerGalleryViewModel = viewModel(graphEntry)
                                val publicPhotographerViewModel: PublicPhotographerViewModel = viewModel()
                                PhotographerPublicProfileScreen(
                                    handle = entry.arguments?.getString("handle"),
                                    viewModel = publicPhotographerViewModel,
                                    onBack = { navController.popBackStack() },
                                    onOpenEvent = { event ->
                                        runnerViewModel.selectEvent(event)
                                        navController.navigate("gallery")
                                    },
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
                                    runnerViewModel = runnerViewModel,
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
                                        navController.navigate("events") {
                                            popUpTo("events") { inclusive = false }
                                            launchSingleTop = true
                                        }
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
                                    onOpenVerify = { navController.navigate("verify") },
                                )
                            }
                                composable("verify") { entry ->
                                    val verifyViewModel: VerifyViewModel = viewModel(entry)
                                    VerifyScreen(
                                        viewModel = verifyViewModel,
                                        onBack = { navController.popBackStack() },
                                        onOpenPhotographer = { handle ->
                                            navController.navigate("photographer/$handle")
                                        },
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
                                    onBrowseEvents = {
                                        navController.navigate("events") {
                                            popUpTo("events") { inclusive = false }
                                            launchSingleTop = true
                                        }
                                    },
                                )
                            }
                            composable(
                                route = RECEIPT_ROUTE_PATTERN,
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
                            onAfterCheckoutSuccess = { order ->
                                // A settled order (₱0/100%-coupon giveaway or a paid replay)
                                // has no PayMongo handoff — land on the receipt so the cart
                                // clears and the downloads show, mirroring the deep-link
                                // return. A null order is the PayMongo bail-back → orders list.
                                val route = order?.let { receiptRoute(it.id, it.returnToken) } ?: "orders"
                                navController.navigate(route) {
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

    private fun handleQuickpitikUri(
        uri: Uri,
        navController: NavController,
        cartViewModel: CartViewModel,
    ) {
        val target = deepLinkTarget(uri) ?: return
        // Email-token links are role-neutral auth surfaces — routed BEFORE the
        // photographer guard below, or a photographer's own verification /
        // email-change link would bounce to studio with the token unredeemed.
        when (target) {
            is DeepLinkTarget.ConfirmEmail -> {
                navController.navigate("confirm-email-change?token=${target.token}") { launchSingleTop = true }
                return
            }
            is DeepLinkTarget.VerifyEmail -> {
                navController.navigate("verify-email?token=${target.token}") { launchSingleTop = true }
                return
            }
            else -> Unit
        }
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
        val overEvents: androidx.navigation.NavOptionsBuilder.() -> Unit = {
            popUpTo("events") { saveState = true }
            launchSingleTop = true
        }
        when (target) {
            is DeepLinkTarget.OrderReturn -> {
                cartViewModel.closeCartSheet()
                cartViewModel.closeCheckoutSheet()
                cartViewModel.resetCheckoutState()
                // F6: a malformed/hostile id falls back to the orders list
                // instead of dead-ending or crashing route parsing.
                val route = receiptRoute(target.orderId, target.token)
                if (route == null) cartViewModel.fetchCart()
                navController.navigate(route ?: "orders", overEvents)
            }
            DeepLinkTarget.Orders -> {
                cartViewModel.closeCartSheet()
                cartViewModel.closeCheckoutSheet()
                cartViewModel.resetCheckoutState()
                cartViewModel.fetchCart()
                navController.navigate("orders", overEvents)
            }
            DeepLinkTarget.Cart -> {
                cartViewModel.closeCheckoutSheet()
                cartViewModel.resetCheckoutState()
                cartViewModel.fetchCart()
                cartViewModel.openCartSheet()
            }
            else -> Unit
        }
    }
}

// Routing decisions lifted out of the Activity so they can be tested without a
// NavHost: cold-start destination, the role guard, the deep-link mapping and
// the receipt route. The Activity keeps only the side effects.

/**
 * A photographer who killed the app while in runner view cold-starts back into
 * runner view — starting in studio with the flag still set would strand the
 * flag, because studio is legal for the true role and the guard would never fire.
 */
internal fun startDestinationFor(signedIn: Boolean, photographer: Boolean, runnerView: Boolean): String = when {
    !signedIn -> "login"
    photographer && runnerView -> "runner"
    photographer -> "studio"
    else -> "runner"
}

/**
 * Where the destination-change guard sends a wrong-role visitor, or null to let
 * them through. Runner routes check the EFFECTIVE role (a photographer in runner
 * view passes); studio routes check the TRUE role (a photographer in runner view
 * may still deep-return to studio). Shared surfaces are in neither set.
 */
internal fun guardRedirect(photographer: Boolean, runnerView: Boolean, route: String): String? {
    val effectiveRunner = !photographer || runnerView
    return when {
        route in RUNNER_ROUTES && !effectiveRunner -> "studio"
        route.startsWith("studio") && !photographer -> "runner"
        else -> null
    }
}

/**
 * The receipt route for an order id plus optional guest RETURN token, or null
 * when the id is not route-safe (F6: the scheme-only deep-link filter is
 * exported, so a hostile id must never be spliced into a route string).
 */
internal fun receiptRoute(orderId: String?, token: String?): String? {
    if (orderId.isNullOrBlank() || !DEEP_LINK_ID.matches(orderId)) return null
    val id = Uri.encode(orderId)
    return if (token.isNullOrBlank()) "orders/return/$id" else "orders/return/$id?token=${Uri.encode(token)}"
}

internal sealed class DeepLinkTarget {
    data class ConfirmEmail(val token: String) : DeepLinkTarget()
    data class VerifyEmail(val token: String) : DeepLinkTarget()
    data class OrderReturn(val orderId: String?, val token: String?) : DeepLinkTarget()
    data object Orders : DeepLinkTarget()
    data object Cart : DeepLinkTarget()
}

/**
 * Only quickpitik:// links and the two https email-token paths the manifest
 * filters on reach the app. startsWith, not ==: the manifest uses pathPrefix,
 * so the OS also delivers "/verify-email/" — an exact match would drop it.
 */
internal fun pickQuickpitikUri(intent: Intent?): Uri? {
    val data = intent?.data ?: return null
    if (data.scheme.equals("quickpitik", ignoreCase = true)) return data
    val path = data.path?.lowercase().orEmpty()
    if ((data.scheme.equals("http", ignoreCase = true) || data.scheme.equals("https", ignoreCase = true)) &&
        (path.startsWith("/verify-email") || path.startsWith("/confirm-email-change"))) {
        return data
    }
    return null
}

/**
 * Maps an accepted link to what the app should do. Host carries the route —
 * Android parses "quickpitik://orders/return" as scheme=quickpitik, host=orders,
 * path=/return. Null = nothing we handle.
 */
internal fun deepLinkTarget(uri: Uri): DeepLinkTarget? {
    val host = uri.host?.lowercase() ?: return null
    val path = uri.path?.lowercase().orEmpty()
    return when {
        host == "confirm-email-change" || path.startsWith("/confirm-email-change") ->
            DeepLinkTarget.ConfirmEmail(uri.getQueryParameter("token").orEmpty())
        host == "verify-email" || path.startsWith("/verify-email") ->
            DeepLinkTarget.VerifyEmail(uri.getQueryParameter("token").orEmpty())
        host == "orders" && path.startsWith("/return") ->
            DeepLinkTarget.OrderReturn(uri.getQueryParameter("orderId"), uri.getQueryParameter("token"))
        host == "orders" -> DeepLinkTarget.Orders
        host == "cart" -> DeepLinkTarget.Cart
        else -> null
    }
}

// The "studio" graph's back-stack entry — the owner of every studio ViewModel.
// They all resolve off this one entry, so they share a lifetime: created lazily
// (a runner session constructs none of them, and none of their init salvos
// fire) and cleared, tether teardown included, exactly when logout pops the
// graph. `remember(entry)` because getBackStackEntry must not run on every
// recomposition.
@Composable
private fun studioGraphEntry(
    navController: NavHostController,
    entry: NavBackStackEntry,
): NavBackStackEntry = remember(entry) { navController.getBackStackEntry("studio") }

@Composable
private fun studioViewModel(
    navController: NavHostController,
    entry: NavBackStackEntry,
): PhotographerDashboardViewModel = viewModel(studioGraphEntry(navController, entry))

@Composable
private fun studioSettingsViewModel(
    navController: NavHostController,
    entry: NavBackStackEntry,
): StudioSettingsViewModel = viewModel(studioGraphEntry(navController, entry))

@Composable
private fun studioEarningsViewModel(
    navController: NavHostController,
    entry: NavBackStackEntry,
): StudioEarningsViewModel = viewModel(studioGraphEntry(navController, entry))

// The tether VM's one cross-concern read is the dashboard's event list, for
// the after-process-death resume (see its KDoc). It is handed over here,
// where both ViewModels already live on the same graph entry, rather than
// being reached for from inside the tether VM. The factory only runs on the
// first resolve; every later call returns that same instance.
@Composable
private fun studioTetherViewModel(
    navController: NavHostController,
    entry: NavBackStackEntry,
): TetherViewModel {
    val graphEntry = studioGraphEntry(navController, entry)
    val dashboard: PhotographerDashboardViewModel = viewModel(graphEntry)
    val application = LocalContext.current.applicationContext as Application
    val factory = remember(dashboard) {
        object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T =
                TetherViewModel(application, dashboard.eventsState) as T
        }
    }
    return viewModel(graphEntry, factory = factory)
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
    val scale by animateFloatAsState(
        targetValue = if (selected) 1.0f else 0.94f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessMediumLow,
        ),
        label = "runnerNavItemScale",
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
