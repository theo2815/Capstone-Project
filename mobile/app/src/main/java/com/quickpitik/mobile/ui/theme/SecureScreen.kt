package com.quickpitik.mobile.ui.theme

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.view.WindowManager
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.window.DialogWindowProvider

// Blocks screenshots, screen recording, and casting for the window this
// composable lives in (Android FLAG_SECURE). Browse-mode surfaces only — the
// runner is looking at an unpurchased, watermarked preview. Owned/purchased
// surfaces never call this.
//
// The lightbox is a Compose Dialog = its own Window, so a flag on the Activity
// window does not cover it; each surface calls this for itself. One holder per
// window; ref-count only if a second same-window caller ever nests.
@Composable
fun SecureScreen() {
    val view = LocalView.current
    if (view.isInEditMode) return
    DisposableEffect(view) {
        val window = (view.parent as? DialogWindowProvider)?.window
            ?: view.context.findActivity()?.window
            ?: return@DisposableEffect onDispose {}
        window.addFlags(WindowManager.LayoutParams.FLAG_SECURE)
        onDispose { window.clearFlags(WindowManager.LayoutParams.FLAG_SECURE) }
    }
}

private fun Context.findActivity(): Activity? =
    generateSequence(this) { (it as? ContextWrapper)?.baseContext }
        .filterIsInstance<Activity>()
        .firstOrNull()
