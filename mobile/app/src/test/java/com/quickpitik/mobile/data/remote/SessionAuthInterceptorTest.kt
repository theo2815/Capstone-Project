package com.quickpitik.mobile.data.remote

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.IOException

@RunWith(RobolectricTestRunner::class)
class SessionAuthInterceptorTest {
    @Test fun `old request cannot be replayed under the new login`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val session = SessionManager.getInstance(context)
        session.saveSession("token-a", "PHOTOGRAPHER", "A", "a@example.com")
        val snapshot = requireNotNull(session.authSnapshot("token-a"))
        session.saveSession("token-b", "PHOTOGRAPHER", "B", "b@example.com")
        assertFalse(session.isCurrent(snapshot))
        assertFalse(session.refreshIfCurrent(snapshot, "late-token-a", "refresh-a"))
        assertFalse(session.clearIfCurrent(snapshot))
        assertEquals("token-b", session.getAccessToken())
        MockWebServer().use { server ->
            server.start()
            val client = OkHttpClient.Builder().addInterceptor(SessionAuthInterceptor(session)).build()
            val request = Request.Builder().url(server.url("/upload")).header("Authorization", "Bearer token-a").build()
            assertThrows(IOException::class.java) { client.newCall(request).execute().close() }
            assertEquals(0, server.requestCount)
        }
    }

    @Test fun `refresh within one login retains ownership of in flight requests`() {
        val session = SessionManager.getInstance(ApplicationProvider.getApplicationContext())
        session.saveSession("token-a", "PHOTOGRAPHER", "A", "a@example.com")
        val snapshot = requireNotNull(session.authSnapshot("token-a"))
        assertTrue(session.refreshIfCurrent(snapshot, "token-a2", "refresh-a2"))
        assertTrue(session.isCurrent(requireNotNull(session.authSnapshot("token-a"))))
        MockWebServer().use { server ->
            server.enqueue(MockResponse().setResponseCode(200))
            server.start()
            val client = OkHttpClient.Builder().addInterceptor(SessionAuthInterceptor(session)).build()
            client.newCall(Request.Builder().url(server.url("/upload")).header("Authorization", "Bearer token-a2").build()).execute().close()
            assertEquals(1, server.requestCount)
        }
    }
}
