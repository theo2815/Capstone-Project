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
import com.quickpitik.mobile.ui.auth.AuthViewModel
import com.quickpitik.mobile.ui.auth.LoginScreen
import com.quickpitik.mobile.ui.auth.RegisterScreen
import com.quickpitik.mobile.ui.photographer.PhotographerDashboardScreen
import com.quickpitik.mobile.ui.photographer.PhotographerDashboardViewModel
import com.quickpitik.mobile.ui.runner.EventsDiscoveryScreen
import com.quickpitik.mobile.ui.runner.FloatingCart
import com.quickpitik.mobile.ui.runner.RunnerGalleryScreen
import com.quickpitik.mobile.ui.runner.RunnerGalleryViewModel
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

                Box(modifier = Modifier.fillMaxSize()) {
                NavHost(
                    navController = navController,
                    startDestination = "login"
                ) {
                    composable("login") {
                        LoginScreen(
                            viewModel = authViewModel,
                            onNavigateToRegister = {
                                navController.navigate("register")
                            },
                            onLoginSuccess = { isPhotographer ->
                                val target = if (isPhotographer) "dashboard" else "events"
                                navController.navigate(target) {
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
                                authViewModel.resetState()
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
                            onEventSelected = { event ->
                                runnerViewModel.selectEvent(event)
                                navController.navigate("gallery")
                            },
                            onNavigateToOrders = { navController.navigate("orders") },
                            onNavigateToProfile = { navController.navigate("profile") },
                            onNavigateToSettings = { navController.navigate("settings") },
                            onLogout = {
                                authViewModel.resetState()
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
                            onNavigateToOrders = {
                                navController.navigate("orders")
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
                                authViewModel.resetState()
                                cartViewModel.clearCart()
                                navController.navigate("login") {
                                    popUpTo("events") { inclusive = true }
                                }
                            }
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
                                authViewModel.resetState()
                                cartViewModel.clearCart()
                                navController.navigate("login") {
                                    popUpTo("events") { inclusive = true }
                                }
                            }
                        )
                    }
                    composable("orders") {
                        OrdersScreen(
                            viewModel = cartViewModel,
                            onNavigateBack = {
                                navController.popBackStack()
                            }
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
