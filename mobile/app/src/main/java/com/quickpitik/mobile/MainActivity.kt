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
import com.quickpitik.mobile.ui.runner.RunnerGalleryScreen
import com.quickpitik.mobile.ui.theme.QuickPitikMobileTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            QuickPitikMobileTheme {
                val navController = rememberNavController()
                val authViewModel: AuthViewModel = viewModel()

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
                                val target = if (isPhotographer) "dashboard" else "gallery"
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
                                val target = if (isPhotographer) "dashboard" else "gallery"
                                navController.navigate(target) {
                                    popUpTo("login") { inclusive = true }
                                }
                            }
                        )
                    }
                    composable("dashboard") {
                        PhotographerDashboardScreen(
                            onLogout = {
                                navController.navigate("login") {
                                    popUpTo("dashboard") { inclusive = true }
                                }
                            }
                        )
                    }
                    composable("gallery") {
                        RunnerGalleryScreen(
                            onLogout = {
                                navController.navigate("login") {
                                    popUpTo("gallery") { inclusive = true }
                                }
                            }
                        )
                    }
                }
            }
        }
    }
}