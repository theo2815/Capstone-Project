package com.quickpitik.mobile.data.local

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Photographer→Runner "view switch" — the mobile port of the website's
 * client-only effective-role pattern (useViewModeStore + useEffectiveRole,
 * shipped web-side 2026-08-26). A PHOTOGRAPHER account can browse the runner
 * surfaces; the TRUE role never changes, so studio access and server-side
 * gates are unaffected. RUNNER-role-gated affordances (cart, saves, photo
 * alerts, runner inbox) are hidden while switched — the backend would 403 a
 * photographer token, and the website leaves them absent too.
 *
 * An object with a StateFlow (the SessionEvents precedent) rather than a
 * threaded callback: the flag is read by MainActivity's route guard, the
 * bottom-bar gate, and RunnerTopBar — a per-screen parameter would touch six
 * hosts for one boolean. Persisted via SessionManager so the choice survives
 * process death (web parity: zustand-persist); login always resets to
 * photographer view (web parity: resetUserScopedStores).
 */
object ViewMode {
    private val _runnerView = MutableStateFlow(false)
    val runnerView: StateFlow<Boolean> = _runnerView

    // RunnerTopBar's "Switch to photographer" item emits here; MainActivity
    // collects and performs the navigation. Same shape as
    // SessionEvents.forcedLogout, for the same reason: the bar has no
    // NavController and threading one through every host screen is churn.
    private val _switchToPhotographer = MutableSharedFlow<Unit>(extraBufferCapacity = 1)
    val switchToPhotographer: SharedFlow<Unit> = _switchToPhotographer

    /** Hydrate the in-memory flag from prefs. Called once at cold start. */
    fun init(session: SessionManager) {
        _runnerView.value = session.isRunnerView()
    }

    fun set(session: SessionManager, runner: Boolean) {
        session.setRunnerView(runner)
        _runnerView.value = runner
    }

    /** Back to photographer view. Also called on every auth transition —
     *  login/logout must always land in the true role's home. */
    fun reset(session: SessionManager) = set(session, false)

    fun requestSwitchToPhotographer() {
        _switchToPhotographer.tryEmit(Unit)
    }
}
