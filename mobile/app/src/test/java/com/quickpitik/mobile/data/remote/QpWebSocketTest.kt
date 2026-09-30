package com.quickpitik.mobile.data.remote

import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestCoroutineScheduler
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit

/**
 * The socket wrapper behind all three live channels, against a real
 * MockWebServer upgrade: the handshake carries the CURRENT token on every
 * attempt (a captured 15-minute JWT would fail forever, invisibly), a
 * signed-out socket never dials, handshake failures back off along the
 * website's capped curve and a successful open resets the count, and the
 * subscriber count keeps the channel alive across Compose Navigation's
 * unordered dispose/compose. Reconnect delays run in virtual time on the
 * injected scope; the short sleeps only wait for the local server.
 */
@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
class QpWebSocketTest {
    @get:Rule val backend = MockBackend()

    private val scheduler = TestCoroutineScheduler()
    private val scope = CoroutineScope(StandardTestDispatcher(scheduler))
    private lateinit var session: SessionManager
    private lateinit var socket: QpWebSocket
    private val serverSockets = CopyOnWriteArrayList<WebSocket>()
    private val opens = Semaphore(0)
    private val clientCloses = CopyOnWriteArrayList<Int>()
    private val delayedCloses = CopyOnWriteArrayList<WebSocket>()
    @Volatile private var accept = true
    @Volatile private var delayClose = false

    @Before fun setUp() {
        session = SessionManager.getInstance(ApplicationProvider.getApplicationContext()).also { it.clearSession() }
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse =
                if (!accept) MockResponse().setResponseCode(401)
                else MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
                    override fun onOpen(webSocket: WebSocket, response: Response) {
                        serverSockets += webSocket
                        opens.release()
                    }
                    override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                        clientCloses += code
                        if (delayClose) delayedCloses += webSocket else webSocket.close(code, reason)
                    }
                })
        }
        socket = QpWebSocket(session, scope)
    }

    @After fun tearDown() {
        socket.release()
        session.clearSession()
        scope.cancel()
    }

    private fun signIn(token: String) = session.saveSession(token, "RUNNER", "Juan", "juan@x.com")
    private fun awaitServerOpen() = assertTrue("server never saw the upgrade", opens.tryAcquire(5, TimeUnit.SECONDS))
    private fun handshake(): RecordedRequest = requireNotNull(backend.server.takeRequest(5, TimeUnit.SECONDS))

    // Runs the pending reconnect, its delay in virtual time, until the server has seen [count] handshakes.
    private fun driveReconnect(count: Int) {
        val deadline = System.currentTimeMillis() + 5_000
        while (backend.server.requestCount < count && System.currentTimeMillis() < deadline) {
            scheduler.advanceUntilIdle()
            Thread.sleep(5)
        }
        assertEquals(count, backend.server.requestCount)
    }

    @Test fun `every attempt re-reads the token, so a refresh between attempts reaches the next handshake`() {
        signIn("tok-a")
        accept = false
        socket.connect(PATH)
        socket.state.awaitFirst { it == WsState.Failed(1) }
        assertEquals("tok-a", handshake().getHeader("Sec-WebSocket-Protocol"))

        session.updateTokens("tok-b", null)
        accept = true
        driveReconnect(2)
        awaitServerOpen()
        val second = handshake()
        assertEquals(PATH, second.path)
        assertEquals("tok-b", second.getHeader("Sec-WebSocket-Protocol"))
        socket.state.awaitFirst { it is WsState.Open }

        val frame = CompletableDeferred<String>()
        val collector = CoroutineScope(Dispatchers.Unconfined).launch { frame.complete(socket.frames.first()) }
        serverSockets.last().send("""{"type":"message.created"}""")
        assertEquals("""{"type":"message.created"}""", runBlocking { withTimeout(5_000) { frame.await() } })
        collector.cancel()
    }

    @Test fun `a signed-out socket stays idle and dials nothing`() {
        socket.connect(PATH)
        assertEquals(WsState.Idle, socket.state.value)
        scheduler.advanceUntilIdle()
        assertEquals(0, backend.server.requestCount)
    }

    @Test fun `handshake rejections back off along the capped curve and a successful open resets the count`() {
        assertEquals(listOf(2000L, 4000L, 8000L, 16000L, 30000L, 30000L, 30000L), (1..7).map { socket.backoffMs(it) })

        signIn("tok-a")
        accept = false
        socket.connect(PATH)
        socket.state.awaitFirst { it == WsState.Failed(1) }
        driveReconnect(2)
        socket.state.awaitFirst { it == WsState.Failed(2) }

        accept = true
        driveReconnect(3)
        awaitServerOpen()
        socket.state.awaitFirst { it is WsState.Open }
        backend.server.shutdown() // the network drops every open connection
        assertEquals(WsState.Failed(1), socket.state.awaitFirst { it is WsState.Failed })
    }

    @Test fun `two subscribers keep the socket open until the last one lets go, and a caller close never reconnects`() {
        signIn("tok-a")
        socket.connect(PATH)
        socket.connect(PATH)
        awaitServerOpen()
        socket.state.awaitFirst { it is WsState.Open }
        assertEquals(1, backend.server.requestCount)

        socket.close()
        Thread.sleep(200)
        assertTrue(clientCloses.isEmpty())
        assertEquals(WsState.Open, socket.state.value)

        socket.close()
        assertEquals(WsState.Idle, socket.state.value)
        val deadline = System.currentTimeMillis() + 5_000
        while (clientCloses.isEmpty() && System.currentTimeMillis() < deadline) Thread.sleep(5)
        assertEquals(listOf(1000), clientCloses)
        repeat(3) { scheduler.advanceUntilIdle(); Thread.sleep(20) }
        assertEquals(1, backend.server.requestCount) // no reconnect after a caller close

        socket.connect(PATH) // a later subscriber dials fresh
        awaitServerOpen()
        assertEquals(2, backend.server.requestCount)
    }

    @Test fun `late close from released socket cannot replace the new connection`() {
        signIn("tok-a")
        delayClose = true
        try {
            socket.connect(PATH)
            awaitServerOpen()
            socket.state.awaitFirst { it is WsState.Open }

            socket.release()
            socket.connect(PATH)
            awaitServerOpen()
            socket.state.awaitFirst { it is WsState.Open }

            val deadline = System.currentTimeMillis() + 5_000
            while (delayedCloses.isEmpty() && System.currentTimeMillis() < deadline) Thread.sleep(5)
            assertEquals(1, delayedCloses.size)
            delayedCloses.single().close(1000, null)
            val failure = runBlocking { withTimeoutOrNull(1_000) { socket.state.first { it is WsState.Failed } } }
            assertEquals(null, failure)
            assertEquals(WsState.Open, socket.state.value)
        } finally {
            delayClose = false
            delayedCloses.forEach { it.close(1000, null) }
        }
    }

    private companion object {
        const val PATH = "/ws/me/runner/notifications"
    }
}
