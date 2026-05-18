package com.quickpitik.mobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.quickpitik.mobile.ui.auth.LoginScreen
import com.quickpitik.mobile.ui.auth.RegisterScreen
import com.quickpitik.mobile.ui.photographer.PhotographerDashboardScreen
import com.quickpitik.mobile.ui.theme.QuickPitikMobileTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            QuickPitikMobileTheme {
                val navController = rememberNavController()

                NavHost(
                    navController = navController,
                    startDestination = "login"
                ) {
                    composable("login") {
                        LoginScreen(
                            onNavigateToRegister = {
                                navController.navigate("register")
                            },
                            onLoginSuccess = {
                                navController.navigate("dashboard") {
                                    popUpTo("login") { inclusive = true }
                                }
                            }
                        )
                    }
                    composable("register") {
                        RegisterScreen(
                            onNavigateToLogin = {
                                navController.navigate("login")
                            },
                            onRegisterSuccess = {
                                navController.navigate("dashboard") {
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
                }
            }
        }
    }
}