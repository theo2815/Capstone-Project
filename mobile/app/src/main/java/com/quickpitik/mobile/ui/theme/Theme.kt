package com.quickpitik.mobile.ui.theme

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val DarkColorScheme = darkColorScheme(
    primary = Fresh,
    onPrimary = Ink,
    primaryContainer = FreshDeep,
    onPrimaryContainer = Color.White,
    secondary = BoneDeep,
    onSecondary = Ink,
    background = Ink,
    onBackground = Bone,
    surface = InkSoft,
    onSurface = Bone,
    surfaceVariant = InkSoft,
    onSurfaceVariant = BoneDeep,
    outline = Slate,
    error = ErrorRed,
    onError = Color.White
)

private val LightColorScheme = lightColorScheme(
    primary = Fresh,
    onPrimary = Color.White,
    primaryContainer = FreshDeep,
    onPrimaryContainer = Color.White,
    secondary = InkSoft,
    onSecondary = Color.White,
    background = Bone,
    onBackground = Ink,
    surface = SurfaceWhite,
    onSurface = Ink,
    surfaceVariant = BoneDeep,
    onSurfaceVariant = InkSoft,
    outline = Line,
    error = ErrorRed,
    onError = Color.White
)

@Composable
fun QuickPitikMobileTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    // We disable dynamicColor by default to enforce the strict "Quiet Studio" brand identity (Ink & Fresh)
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    // Lock in the custom design identity matching the website
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            val insetsController = WindowCompat.getInsetsController(window, view)
            // Black status bar icons when theme is light, white status bar icons when theme is dark
            insetsController.isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}