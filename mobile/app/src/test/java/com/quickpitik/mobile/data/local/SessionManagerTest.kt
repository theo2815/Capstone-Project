package com.quickpitik.mobile.data.local

import android.app.Application
import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.service.TetherIngestService
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf

/**
 * The session store's account boundaries: a fresh login starts a new auth
 * generation, forgets the previous login's in-flight tokens and its runner-view
 * choice; a refresh keeps a short ring of recent tokens so requests already in
 * flight still pass the interceptor; clearing drops everything an account
 * owns, including a running tether watch. (Robolectric has no Android keystore,
 * so these run on the memory-token fallback; the keystore-unavailable branch
 * itself is asserted in PayoutSecurityTest.)
 */
@RunWith(RobolectricTestRunner::class)
class SessionManagerTest {
    private val context: Context = ApplicationProvider.getApplicationContext()
    // A fresh instance bound to THIS test's Application: the getInstance singleton
    // outlives Robolectric's per-test Application, so its prefs handle goes stale.
    private val session: SessionManager = SessionManager::class.java.getDeclaredConstructor(Context::class.java)
        .apply { isAccessible = true }.newInstance(context)

    @Before fun fresh() = session.clearSession()
    @After fun clear() = session.clearSession()

    @Test fun `the role check is case-insensitive and photographer-or-not`() {
        assertTrue(isPhotographerRole("PHOTOGRAPHER"))
        assertTrue(isPhotographerRole("photographer"))
        assertFalse(isPhotographerRole("RUNNER"))
        assertFalse(isPhotographerRole("ADMIN"))
        assertFalse(isPhotographerRole(null))
    }

    @Test fun `a fresh login starts a new generation, forgets in-flight tokens and lands in the true role's view`() {
        session.saveSession("t0", "PHOTOGRAPHER", "Juan", "juan@x.com", refreshToken = "r0")
        session.updateTokens("t1", null)
        val inFlight = requireNotNull(session.authSnapshot("t0")) // the old token is still in the ring
        session.setRunnerView(true)

        session.saveSession("n0", "PHOTOGRAPHER", "Juan", "juan@x.com", refreshToken = "r1")
        assertFalse(session.isCurrent(inFlight))
        assertNull(session.authSnapshot("t0"))
        assertNull(session.authSnapshot("t1"))
        assertNotNull(session.authSnapshot("n0"))
        assertFalse(session.isRunnerView())
    }

    @Test fun `token refreshes keep the eight most recent tokens usable for requests already in flight`() {
        session.saveSession("t0", "RUNNER", "Juan", "juan@x.com", refreshToken = "r")
        for (i in 1..9) session.updateTokens("t$i", null)
        assertEquals("t9", session.getAccessToken())
        assertNull(session.authSnapshot("t0"))
        for (i in 1..9) assertNotNull("t$i", session.authSnapshot("t$i"))
    }

    @Test fun `clearing the session drops the tokens, the tether watch and stops the ingest service`() {
        session.saveSession("t0", "PHOTOGRAPHER", "Juan", "juan@x.com", refreshToken = "r0")
        val tether = context.getSharedPreferences("quickpitik_tether", Context.MODE_PRIVATE)
        tether.edit()
            .putString("watch_active_event", "e1")
            .putString("watch_owner", "juan@x.com")
            .putString("watch_event_snapshot", "{}")
            .putBoolean("keep_me", true)
            .commit()
        val live = requireNotNull(session.authSnapshot("t0"))
        val shadow = shadowOf(context as Application)
        while (shadow.nextStoppedService != null) { /* drain the @Before clear */ }

        session.clearSession()
        assertNull(session.getAccessToken())
        assertNull(session.getRefreshToken())
        assertNull(session.getUserEmail())
        assertNull(session.getUserRole())
        assertFalse(session.isCurrent(live))
        assertNull(session.authSnapshot("t0"))
        assertNull(tether.getString("watch_active_event", null))
        assertNull(tether.getString("watch_owner", null))
        assertNull(tether.getString("watch_event_snapshot", null))
        assertTrue(tether.getBoolean("keep_me", false)) // only the watch keys go
        assertEquals(TetherIngestService::class.java.name, shadow.nextStoppedService?.component?.className)
    }
}
