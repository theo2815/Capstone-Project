package com.quickpitik.mobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.quickpitik.mobile.ui.auth.AuthViewModel
import com.quickpitik.mobile.ui.auth.LoginScreen
import com.quickpitik.mobile.ui.auth.RegisterScreen
import com.quickpitik.mobile.ui.photographer.PhotographerDashboardScreen
import com.quickpitik.mobile.ui.photographer.PhotographerDashboardViewModel
import com.quickpitik.mobile.ui.runner.EventsDiscoveryScreen
import com.quickpitik.mobile.ui.runner.RunnerGalleryScreen
import com.quickpitik.mobile.ui.runner.RunnerGalleryViewModel
import com.quickpitik.mobile.ui.runner.SavedEventsViewModel
import com.quickpitik.mobile.ui.runner.CartViewModel
import com.quickpitik.mobile.ui.runner.CartScreen
import com.quickpitik.mobile.ui.runner.CheckoutScreen
import com.quickpitik.mobile.ui.runner.OrdersScreen
import com.quickpitik.mobile.ui.runner.ProfileViewModel
import com.quickpitik.mobile.ui.runner.ProfileScreen
import com.quickpitik.mobile.ui.runner.AccountSettingsScreen
import com.quickpitik.mobile.ui.theme.QuickPitikMobileTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            QuickPitikMobileTheme {
                val navController = rememberNavController()
                val authViewModel: AuthViewModel = viewModel()
                val cartViewModel: CartViewModel = viewModel()
                val profileViewModel: ProfileViewModel = viewModel()
                // Hoisted to the NavHost scope so the events-discovery browse screen,
                // the gallery cockpit, and the profile race log all read/write the one
                // shared instance (selected event + saved-events store stay in sync).
                val runnerViewModel: RunnerGalleryViewModel = viewModel()
                val savedEventsViewModel: SavedEventsViewModel = viewModel()

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
                            cartViewModel = cartViewModel,
                            savedEventsViewModel = savedEventsViewModel,
                            onEventSelected = { event ->
                                runnerViewModel.selectEvent(event)
                                navController.navigate("gallery")
                            },
                            onNavigateToCart = { navController.navigate("cart") },
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
                            onNavigateToCart = {
                                navController.navigate("cart")
                            },
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
                    composable("cart") {
                        CartScreen(
                            viewModel = cartViewModel,
                            onNavigateBack = {
                                navController.popBackStack()
                            },
                            onNavigateToCheckout = {
                                navController.navigate("checkout")
                            }
                        )
                    }
                    composable("checkout") {
                        CheckoutScreen(
                            viewModel = cartViewModel,
                            onNavigateBack = {
                                navController.popBackStack()
                            },
                            onCheckoutSuccess = {
                                navController.navigate("orders") {
                                    popUpTo("cart") { inclusive = true }
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
                }
            }
        }
    }
}
