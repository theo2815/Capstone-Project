package com.quickpitik.mobile.ui.theme

import android.app.Activity
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.TextStyle
import androidx.core.view.WindowCompat

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
    // We disable dynamicColor by default to enforce the strict Finish Line
    // brand identity (Ink & Fresh)
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    // Light-only, deliberately: the design system paints Bone/Ink directly in
    // every screen, so the old system-driven dark scheme only ever changed the
    // status bar — black bar with light icons over cream content. A real dark
    // mode would need per-screen token work; until then the app pins light.
    val colorScheme = LightColorScheme

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            val insetsController = WindowCompat.getInsetsController(window, view)
            // Bone background → dark status-bar icons.
            insetsController.isAppearanceLightStatusBars = true
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography
    ) {
        // Default every bare Text() to the Archivo body font so the website type
        // shows app-wide without touching each call site. Explicit
        // `style = Typography.label*/title*` still wins (mono kickers, display
        // headings); this only sets the fallback family.
        CompositionLocalProvider(
            LocalTextStyle provides LocalTextStyle.current.merge(
                TextStyle(fontFamily = BodyFontFamily)
            ),
            content = content
        )
    }
}