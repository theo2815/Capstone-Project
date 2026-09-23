package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
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
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * The runner inbox's merge rules over a real socket: every (re)open pulls the
 * authoritative list, a push lands newest-first and is deduped against it,
 * frames of any other type are dropped, a dead network only explains an empty
 * inbox, and a read mark refetches so the bell follows the server.
 */
@RunWith(RobolectricTestRunner::class)
class RunnerInboxViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private val serverSockets = CopyOnWriteArrayList<WebSocket>()
    private val opens = Semaphore(0)
    private val gets = AtomicInteger()
    @Volatile private var list: MockResponse = ok(TWO)

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("RUNNER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val path = request.path.orEmpty()
                return when {
                    path == "/ws/me/runner/notifications" -> MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
                        override fun onOpen(webSocket: WebSocket, response: Response) {
                            serverSockets += webSocket
                            opens.release()
                        }
                        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                            webSocket.close(code, reason) // complete the close handshake so the server can shut down
                        }
                    })
                    path == "/api/v1/me/runner/messages" && request.method == "GET" -> {
                        gets.incrementAndGet()
                        list
                    }
                    path == "/api/v1/me/runner/messages/m2/read" -> {
                        list = ok(ALL_READ)
                        ok(message("m2", read = true))
                    }
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }
    }

    @After fun tearDown() = SessionManager.getInstance(application).clearSession()

    private fun awaitUntil(check: () -> Boolean) {
        val deadline = System.currentTimeMillis() + 5_000
        while (!check() && System.currentTimeMillis() < deadline) Thread.sleep(5)
        assertTrue(check())
    }

    @Test fun `every open pulls the list, pushes merge newest-first and dedupe, other frames are ignored`() {
        val vm = RunnerInboxViewModel(application)
        vm.connect()
        assertTrue(opens.tryAcquire(5, TimeUnit.SECONDS))
        vm.messages.awaitFirst { it.size == 2 }
        assertEquals(1, gets.get())
        assertEquals(1, vm.unreadCount.awaitFirst { it == 1 })

        val server = serverSockets.last()
        server.send(frame("m3"))
        vm.messages.awaitFirst { it.size == 3 }
        server.send(frame("m3")) // re-broadcast
        server.send("""{"type":"photo.published"}""") // another channel's type
        server.send("not json")
        server.send(frame("m4"))
        vm.messages.awaitFirst { it.size == 4 }
        assertEquals(listOf("m4", "m3", "m1", "m2"), vm.messages.value.map { it.id })
        assertEquals(3, vm.unreadCount.awaitFirst { it == 3 })
        vm.disconnect()
    }

    @Test fun `a dead network explains an empty inbox but never hides cached messages`() {
        list = MockResponse().setResponseCode(500)
        val vm = RunnerInboxViewModel(application)
        vm.fetchMessages()
        assertEquals("HTTP 500", vm.fetchError.awaitFirst { it != null })

        list = ok(TWO)
        vm.fetchMessages()
        vm.messages.awaitFirst { it.size == 2 }
        vm.fetchError.awaitFirst { it == null }

        list = MockResponse().setResponseCode(500)
        vm.fetchMessages()
        awaitUntil { gets.get() == 3 }
        Thread.sleep(50)
        assertEquals(2, vm.messages.value.size)
        assertNull(vm.fetchError.value)
    }

    @Test fun `marking a message read refetches so the badge follows the server`() {
        val vm = RunnerInboxViewModel(application)
        vm.fetchMessages()
        assertEquals(1, vm.unreadCount.awaitFirst { it == 1 })
        vm.markRead("m2")
        assertEquals(0, vm.unreadCount.awaitFirst { it == 0 })
        assertEquals(2, gets.get())
    }

    private companion object {
        fun message(id: String, read: Boolean) =
            """{"id":"$id","kind":"refund_decision","title":null,"body":"Refund approved","orderId":"o1","sourceDecisionId":null,"createdAt":"2026-09-11T00:00:00Z","readAt":${if (read) "\"2026-09-11T01:00:00Z\"" else "null"}}"""
        fun frame(id: String) = """{"type":"message.created","message":${message(id, read = false)}}"""
        val TWO = "[${message("m1", read = true)},${message("m2", read = false)}]"
        val ALL_READ = "[${message("m1", read = true)},${message("m2", read = true)}]"
    }
}
