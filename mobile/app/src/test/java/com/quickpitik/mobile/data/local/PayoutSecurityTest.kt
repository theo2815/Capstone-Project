package com.quickpitik.mobile.data.local

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.ui.photographer.isFutureIso
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.time.Instant
import java.util.TimeZone

@RunWith(RobolectricTestRunner::class)
@Config(application = android.app.Application::class)
class PayoutSecurityTest {
    @Test fun cooldownRespectsUtcAndOffsetTimestampsInCebu() {
        val original = TimeZone.getDefault()
        try {
            TimeZone.setDefault(TimeZone.getTimeZone("Asia/Manila"))
            val future = Instant.now().plusSeconds(3600)
            assertTrue(isFutureIso(future.toString()))
            assertTrue(isFutureIso(future.atOffset(java.time.ZoneOffset.ofHours(8)).toString()))
            assertFalse(isFutureIso(Instant.now().minusSeconds(3600).toString()))
            assertFalse(isFutureIso("invalid"))
        } finally {
            TimeZone.setDefault(original)
        }
    }

    @Test fun unavailableKeystoreKeepsTokensOnlyInMemoryAndRestartRequiresLogin() {
        // Robolectric supplies no AndroidKeyStore provider, exercising the real catch branch.
        assertNull(java.security.Security.getProvider("AndroidKeyStore"))
        val context = ApplicationProvider.getApplicationContext<Context>()
        val constructor = SessionManager::class.java.getDeclaredConstructor(Context::class.java)
            .apply { isAccessible = true }
        val session = constructor.newInstance(context)
        session.saveSession("review-access", "PHOTOGRAPHER", "Review", "review@test.local",
            refreshToken = "review-refresh")
        val plaintext = context.getSharedPreferences("quickpitik_session", Context.MODE_PRIVATE)
        assertNull(plaintext.getString("access_token", null))
        assertNull(plaintext.getString("refresh_token", null))
        assertEquals("review-access", session.getAccessToken())
        session.updateTokens("rotated-access", "rotated-refresh")
        assertEquals("rotated-access", session.getAccessToken())
        assertEquals("rotated-refresh", session.getRefreshToken())
        assertNull(plaintext.getString("access_token", null))
        assertNull(plaintext.getString("refresh_token", null))
        assertNull(constructor.newInstance(context).getAccessToken())
        session.clearSession()
        assertNull(session.getRefreshToken())
    }
}
