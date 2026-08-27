package com.quickpitik.mobile.ui.runner

import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.isPhotographerRole

/**
 * TRUE-role check for runner surfaces under the photographer→runner view
 * switch (see ViewMode). A PHOTOGRAPHER browsing in runner view fails this,
 * and every RUNNER-role-gated affordance — cart, saved events, photo alerts,
 * the runner inbox — must be HIDDEN for them (the backend 403s a photographer
 * token on those endpoints; the website leaves them absent too, its
 * "degrade to empty rather than 403" contract). Everything else on runner
 * surfaces (browsing, galleries, profile/selfies/account) is Bearer-gated and
 * stays live for both.
 *
 * Plain remember{}: the role only changes across login, which recreates the
 * whole composition.
 */
@Composable
fun rememberIsTrueRunner(): Boolean {
    val context = LocalContext.current
    // Read the role LIVE — no keyless remember{}. Login only navigates; it does
    // NOT recreate MainActivity's composition, so a cached value goes stale after
    // an in-process role change (photographer → sign out → runner login). The
    // root-mounted FloatingCart never leaves composition, so a stale cache there
    // left a real runner with no cart pill even though the freshly-composed
    // gallery still let them add. A SharedPreferences read per recomposition is
    // cheap (in-memory after first load) and always current.
    return !isPhotographerRole(SessionManager.getInstance(context).getUserRole())
}
