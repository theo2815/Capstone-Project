package com.quickpitik.mobile.data.remote

import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionEvents
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.ok
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.QueueDispatcher
import okhttp3.mockwebserver.RecordedRequest
import okhttp3.mockwebserver.SocketPolicy
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

/**
 * The 401 → refresh → retry ladder that keeps a photographer signed in when the
 * 15-minute access token lapses mid-shoot. An expiry costs one refresh, even
 * when parallel uploads all 401 at once (the backend rotates refresh tokens, so
 * a second refresh would sign the phone out); the retry carries the new bearer;
 * only a refused refresh token signs out, and only a suspension says why; a
 * refresh that never gets an answer keeps the session; /auth/ calls and an
 * already-retried call never refresh. A real OkHttp client wired like
 * RetrofitClient's, against MockWebServer; the refresh itself goes through
 * RetrofitClient.refreshApi, which follows the same base URL. Refresh answers of
 * 429/500/503 are covered in SessionAuthInterceptorTest.
 */
@RunWith(RobolectricTestRunner::class)
class TokenAuthenticatorTest {
    @get:Rule val backend = MockBackend()

    private val session = SessionManager.getInstance(ApplicationProvider.getApplicationContext())
    private val client = OkHttpClient.Builder()
        .addInterceptor(SessionAuthInterceptor(session))
        .authenticator(TokenAuthenticator(session))
        .build()

    @Before fun setUp() {
        signIn(refreshToken = "ref-a")
        // A request no test planned for (a refresh that should not happen) gets a 404, not a hang.
        (backend.server.dispatcher as QueueDispatcher).setFailFast(true)
    }

    @After fun tearDown() = session.clearSession()

    private fun signIn(refreshToken: String?) =
        session.saveSession("tok-a", "PHOTOGRAPHER", "Juan", "juan@x.com", refreshToken = refreshToken)

    /** A call carrying the login's bearer; returns the status its caller finally sees. */
    private fun call(path: String = "/api/v1/me/photographer/events"): Int =
        client.newCall(Request.Builder().url(backend.server.url(path)).header("Authorization", "Bearer tok-a").build())
            .execute().use { it.code }

    private fun expired() = MockResponse().setResponseCode(401)
    private fun refreshed(access: String, refresh: String) = ok("""{"accessToken":"$access","refreshToken":"$refresh"}""")
    private fun refused(code: String, message: String) = MockResponse().setResponseCode(401)
        .setBody("""{"success":false,"errors":[{"code":"$code","message":"$message"}]}""")

    /** SessionEvents keeps no replay, so the listener is attached before [block] runs. */
    private fun forcedLogoutsDuring(block: () -> Unit): List<String?> {
        val seen = CopyOnWriteArrayList<String?>()
        val listener = CoroutineScope(Dispatchers.Unconfined).launch { SessionEvents.forcedLogout.collect { seen += it } }
        try { block() } finally { listener.cancel() }
        return seen
    }

    @Test fun `an expired token is refreshed and the call retried with the new bearer`() {
        backend.enqueue(expired(), refreshed("tok-b", "ref-b"), MockResponse())
        val logouts = forcedLogoutsDuring { assertEquals(200, call()) }

        assertEquals("Bearer tok-a", backend.server.takeRequest().getHeader("Authorization"))
        val refresh = backend.server.takeRequest()
        assertEquals("/api/v1/auth/refresh", refresh.path)
        assertEquals("""{"refreshToken":"ref-a"}""", refresh.body.readUtf8())
        assertNull(refresh.getHeader("Authorization")) // the dead bearer never rides along
        assertEquals("Bearer tok-b", backend.server.takeRequest().getHeader("Authorization"))
        assertEquals("tok-b", session.getAccessToken())
        assertEquals("ref-b", session.getRefreshToken())
        assertTrue(logouts.isEmpty())
    }

    @Test fun `a refresh that never gets an answer keeps the session and surfaces the 401`() {
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse =
                if (request.path == "/api/v1/auth/refresh") MockResponse().setSocketPolicy(SocketPolicy.DISCONNECT_AFTER_REQUEST)
                else expired()
        }
        val logouts = forcedLogoutsDuring { assertEquals(401, call()) }
        assertEquals("tok-a", session.getAccessToken())
        assertEquals("ref-a", session.getRefreshToken())
        assertTrue(logouts.isEmpty())
    }

    @Test fun `a refused refresh signs out, and only a suspension says why`() {
        backend.enqueue(expired(), refused("ACCOUNT_SUSPENDED", "Account suspended"))
        assertEquals(listOf("Account suspended"), forcedLogoutsDuring { assertEquals(401, call()) })
        assertNull(session.getAccessToken())
        assertNull(session.getUserEmail())

        signIn(refreshToken = "ref-a")
        backend.enqueue(expired(), refused("INVALID_REFRESH_TOKEN", "Refresh token expired or revoked"))
        assertEquals(listOf<String?>(null), forcedLogoutsDuring { assertEquals(401, call()) })
        assertNull(session.getAccessToken())

        // No refresh token on file: nothing to spend, so it signs out without asking the server.
        signIn(refreshToken = null)
        backend.enqueue(expired())
        assertEquals(listOf<String?>(null), forcedLogoutsDuring { assertEquals(401, call()) })
        assertEquals(5, backend.server.requestCount)
        assertNull(session.getAccessToken())
    }

    @Test fun `a 401 from an auth endpoint never spends the refresh token`() {
        // The bearer is attached so the /auth/ rule is all that stands between this 401 and a refresh.
        backend.enqueue(expired())
        val logouts = forcedLogoutsDuring { assertEquals(401, call("/api/v1/auth/login")) }
        assertEquals(1, backend.server.requestCount)
        assertEquals("tok-a", session.getAccessToken())
        assertTrue(logouts.isEmpty())
    }

    @Test fun `parallel 401s spend the refresh token once`() {
        val refreshes = AtomicInteger()
        val bothExpired = CountDownLatch(2)
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse = when {
                request.path == "/api/v1/auth/refresh" -> {
                    refreshes.incrementAndGet()
                    // Hold the refresh until both calls have their 401, then give the second
                    // one time to reach the authenticator while this refresh is still out.
                    bothExpired.await(5, TimeUnit.SECONDS)
                    Thread.sleep(200)
                    refreshed("tok-b", "ref-b")
                }
                request.getHeader("Authorization") == "Bearer tok-b" -> MockResponse()
                else -> expired().also { bothExpired.countDown() }
            }
        }
        val codes = CopyOnWriteArrayList<Int>()
        List(2) { thread { codes += call() } }.forEach { it.join(10_000) }

        assertEquals(listOf(200, 200), codes)
        assertEquals(1, refreshes.get())
        assertEquals("tok-b", session.getAccessToken())
    }

    @Test fun `a retried call that 401s again gives up instead of refreshing again`() {
        backend.enqueue(expired(), refreshed("tok-b", "ref-b"), expired())
        val logouts = forcedLogoutsDuring { assertEquals(401, call()) }
        assertEquals(3, backend.server.requestCount) // the call, one refresh, one retry
        assertEquals("tok-b", session.getAccessToken()) // the refreshed login stays; only this call fails
        assertTrue(logouts.isEmpty())
    }
}
