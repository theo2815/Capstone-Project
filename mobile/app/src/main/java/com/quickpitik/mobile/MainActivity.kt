package com.quickpitik.mobile

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.quickpitik.mobile.data.local.SessionEvents
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.ui.auth.AuthViewModel
import com.quickpitik.mobile.ui.auth.ForgotPasswordScreen
import com.quickpitik.mobile.ui.auth.LoginScreen
import com.quickpitik.mobile.ui.auth.RegisterScreen
import com.quickpitik.mobile.ui.auth.ResetPasswordScreen
import com.quickpitik.mobile.ui.photographer.PhotographerDashboardScreen
import com.quickpitik.mobile.ui.photographer.PhotographerDashboardViewModel
import com.quickpitik.mobile.ui.photographer.PhotographerPublicProfileScreen
import com.quickpitik.mobile.ui.photographer.PublicPhotographerViewModel
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
                val profileViewModel: ProfileViewModel = viewModel()
                // Hoisted to the NavHost scope so the events-discovery browse screen,
                // the gallery cockpit, and the profile race log all read/write the one
                // shared instance (selected event + saved-events store stay in sync).
                val runnerViewModel: RunnerGalleryViewModel = viewModel()
                val savedEventsViewModel: SavedEventsViewModel = viewModel()
                // Hoisted like savedEventsViewModel so the bell's unread badge is
                // the same number on every runner surface that mounts it.
                val runnerInboxViewModel: RunnerInboxViewModel = viewModel()

                // Cold start with a cached JWT should land on the user's home
                // surface, not bounce them through login again. Same role→route
                // mapping as onLoginSuccess below, kept in one place. remember{}
                // so a later clearSession() can't re-key the NavHost mid-session.
                val sessionManager = remember { SessionManager.getInstance(this@MainActivity) }
                val startDestination = remember {
                    when {
                        sessionManager.getAccessToken() == null -> "login"
                        sessionManager.getUserRole()
                            .equals("PHOTOGRAPHER", ignoreCase = true) -> "dashboard"
                        else -> "events"
                    }
                }

                // Raised by TokenAuthenticator when refresh fails: the session is
                // unrecoverable, so drop the whole back stack and land on login.
                LaunchedEffect(Unit) {
                    SessionEvents.forcedLogout.collect {
                        navController.navigate("login") {
                            popUpTo(0) { inclusive = true }
                            launchSingleTop = true
                        }
                    }
                }

                Box(modifier = Modifier.fillMaxSize()) {
                NavHost(
                    navController = navController,
                    startDestination = startDestination
                ) {
                    composable("login") {
                        LoginScreen(
                            viewModel = authViewModel,
                            onNavigateToRegister = {
                                navController.navigate("register")
                            },
                            onNavigateToForgotPassword = {
                                navController.navigate("forgot-password")
                            },
                            onLoginSuccess = { isPhotographer ->
                                val target = if (isPhotographer) "dashboard" else "events"
                                navController.navigate(target) {
                                    popUpTo("login") { inclusive = true }
                                }
                            }
                        )
                    }
                    // Auth recovery. Both mirror website /(auth)/forgot-password
                    // and /(auth)/reset-password. No nav arguments: the reset
                    // email links to the WEBSITE origin (EmailService builds it
                    // from app.cors.allowed-origins), so the token is pasted on
                    // the reset screen rather than carried in a deep link.
                    // "Back to sign in" replaces the login entry instead of
                    // stacking a second one.
                    composable("forgot-password") {
                        ForgotPasswordScreen(
                            viewModel = authViewModel,
                            onNavigateToLogin = {
                                navController.navigate("login") {
                                    popUpTo("login") { inclusive = true }
                                }
                            },
                            onNavigateToReset = {
                                navController.navigate("reset-password")
                            }
                        )
                    }
                    composable("reset-password") {
                        ResetPasswordScreen(
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
                                val target = if (isPhotographer) "dashboard" else "events"
                                navController.navigate(target) {
                                    popUpTo("login") { inclusive = true }
                                }
                            }
                        )
                    }
                    composable("dashboard") {
                        val photographerViewModel: PhotographerDashboardViewModel = viewModel()
                        PhotographerDashboardScreen(
                            viewModel = photographerViewModel,
                            onLogout = {
                                authViewModel.logout()
                                navController.navigate("login") {
                                    popUpTo("dashboard") { inclusive = true }
                                }
                            }
                        )
                    }
                    // Runner landing — browse every race (web /events), then tap a card
                    // to open its cockpit. Selecting an event seeds the shared
                    // RunnerGalleryViewModel before navigating to the gallery.
                    composable("events") {
                        EventsDiscoveryScreen(
                            viewModel = runnerViewModel,
                            savedEventsViewModel = savedEventsViewModel,
                            inboxViewModel = runnerInboxViewModel,
                            onEventSelected = { event ->
                                runnerViewModel.selectEvent(event)
                                navController.navigate("gallery")
                            },
                            onNavigateToOrders = { navController.navigate("orders") },
                            onNavigateToProfile = { navController.navigate("profile") },
                            onNavigateToSettings = { navController.navigate("settings") },
                            onOpenOrder = { orderId -> navController.navigate("orders?orderId=$orderId") },
                            onLogout = {
                                authViewModel.logout()
                                cartViewModel.clearCart()
                                navController.navigate("login") {
                                    popUpTo("events") { inclusive = true }
                                }
                            }
                        )
                    }
                    composable("gallery") {
                        RunnerGalleryScreen(
                            viewModel = runnerViewModel,
                            cartViewModel = cartViewModel,
                            inboxViewModel = runnerInboxViewModel,
                            onNavigateToOrders = {
                                navController.navigate("orders")
                            },
                            onOpenOrder = { orderId ->
                                navController.navigate("orders?orderId=$orderId")
                            },
                            onOpenPhotographer = { handle ->
                                navController.navigate("photographer/$handle")
                            },
                            onNavigateToProfile = {
                                navController.navigate("profile")
                            },
                            onNavigateToSettings = {
                                navController.navigate("settings")
                            },
                            onNavigateBack = {
                                navController.popBackStack()
                            },
                            onLogout = {
                                authViewModel.logout()
                                cartViewModel.clearCart()
                                navController.navigate("login") {
                                    popUpTo("events") { inclusive = true }
                                }
                            }
                        )
                    }
                    // Public photographer profile — website /{handle}. Reached
                    // from the photo byline in the runner lightbox; the same
                    // screen also serves the photographer's own "Preview public
                    // profile" from inside the dashboard shell. Only ever
                    // navigated to with a non-null handle.
                    composable(
                        route = "photographer/{handle}",
                        arguments = listOf(navArgument("handle") { type = NavType.StringType }),
                    ) { entry ->
                        val publicPhotographerViewModel: PublicPhotographerViewModel = viewModel()
                        PhotographerPublicProfileScreen(
                            handle = entry.arguments?.getString("handle"),
                            viewModel = publicPhotographerViewModel,
                            onBack = { navController.popBackStack() },
                        )
                    }
                    composable("profile") {
                        ProfileScreen(
                            viewModel = profileViewModel,
                            cartViewModel = cartViewModel,
                            savedEventsViewModel = savedEventsViewModel,
                            onNavigateBack = {
                                navController.popBackStack()
                            },
                            onOpenEvent = { slug ->
                                val event = runnerViewModel.eventBySlug(slug)
                                if (event != null) {
                                    runnerViewModel.selectEvent(event)
                                    navController.navigate("gallery")
                                } else {
                                    // Event not in the loaded set — fall back to browse.
                                    navController.navigate("events")
                                }
                            },
                            onBrowseEvents = {
                                navController.navigate("events")
                            }
                        )
                    }
                    composable("settings") {
                        AccountSettingsScreen(
                            viewModel = profileViewModel,
                            onNavigateBack = {
                                navController.popBackStack()
                            },
                            onLogout = {
                                authViewModel.logout()
                                cartViewModel.clearCart()
                                navController.navigate("login") {
                                    popUpTo("events") { inclusive = true }
                                }
                            }
                        )
                    }
                    // Optional orderId arg so the runner inbox can deep-link a
                    // dispute-outcome message straight to that order's detail.
                    // Plain navigate("orders") still matches with a null arg.
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
                            onNavigateBack = {
                                navController.popBackStack()
                            },
                            initialOrderId = entry.arguments?.getString("orderId"),
                        )
                    }
                    // PayMongo return surface — entered via the quickpitik://
                    // deep link MobileReturnController emits after the gateway
                    // bounces the browser back. Reads the orderId arg, polls
                    // /me/orders/{id} until PAID, then renders the receipt
                    // editorial.
                    composable(
                        route = "orders/return/{orderId}",
                        arguments = listOf(navArgument("orderId") { type = NavType.StringType }),
                    ) { entry ->
                        val orderId = entry.arguments?.getString("orderId").orEmpty()
                        OrderReturnScreen(
                            orderId = orderId,
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

                // Global floating-cart pill + cart/checkout sheets — mirrors website
                // FloatingCart. Lives outside the NavHost so it overlays every route.
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
                    navController.navigate("orders/return/$orderId") {
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
