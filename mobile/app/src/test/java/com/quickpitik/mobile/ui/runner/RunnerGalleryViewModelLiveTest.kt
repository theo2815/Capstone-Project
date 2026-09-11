package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.WsState
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
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.time.LocalDate
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit

/**
 * The cockpit's live-photo badge: only a LIVE event opens the channel, only
 * `photo.published` frames count (the indexing frames ride the same channel),
 * the badge tap clears the count with a fresh server page, and the retry
 * affordance starts the socket over instead of waiting out its backoff.
 */
@RunWith(RobolectricTestRunner::class)
class RunnerGalleryViewModelLiveTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private val serverSockets = CopyOnWriteArrayList<WebSocket>()
    private val opens = Semaphore(0)
    private val paths = CopyOnWriteArrayList<String>()

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("RUNNER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val path = request.path.orEmpty()
                paths += path
                return when {
                    path.startsWith("/ws/events/") -> MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
                        override fun onOpen(webSocket: WebSocket, response: Response) {
                            serverSockets += webSocket
                            opens.release()
                        }
                        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                            webSocket.close(code, reason) // complete the close handshake so the server can shut down
                        }
                    })
                    path.startsWith("/api/v1/events/") && path.contains("/photos") ->
                        ok("""{"items":[],"total":0,"offset":0,"limit":60,"snapshotAt":"2026-09-11T00:00:00Z"}""")
                    else -> MockResponse().setResponseCode(404) // the editorial detail is best-effort
                }
            }
        }
    }

    @After fun tearDown() = SessionManager.getInstance(application).clearSession()

    private fun event(date: LocalDate) = EventDto("ev-1", "cebu", "Cebu Run", date.toString(), "Cebu", null, 0, 0, "LIVE")
    private fun photoPages() = paths.count { it.startsWith("/api/v1/events/") && it.contains("/photos") }
    private fun sockets() = paths.count { it.startsWith("/ws/events/ev-1/photos") }
    private fun awaitUntil(check: () -> Boolean) {
        val deadline = System.currentTimeMillis() + 5_000
        while (!check() && System.currentTimeMillis() < deadline) Thread.sleep(5)
        assertTrue(check())
    }

    @Test fun `published pushes bump the badge, other frames do not, refresh clears it with a fresh page, retry restarts the socket`() {
        val vm = RunnerGalleryViewModel(application)
        vm.selectEvent(event(LocalDate.now()))
        vm.searchState.awaitFirst { it is PhotosSearchState.Success }
        vm.connectLivePhotos()
        assertTrue(opens.tryAcquire(5, TimeUnit.SECONDS))
        assertEquals(1, sockets())
        vm.liveState.awaitFirst { it is WsState.Open }

        val server = serverSockets.last()
        server.send("""{"type":"photo.indexed","id":"p1","indexingStatus":"done"}""")
        server.send("""{"type":"photo.published","id":"p1"}""")
        server.send("""{"type":"photo.published","id":"p2"}""")
        assertEquals(2, vm.newPhotoCount.awaitFirst { it == 2 })

        val pagesBefore = photoPages()
        vm.refreshLivePhotos()
        assertEquals(0, vm.newPhotoCount.value)
        awaitUntil { photoPages() == pagesBefore + 1 }

        vm.retryLivePhotos()
        assertTrue(opens.tryAcquire(5, TimeUnit.SECONDS))
        assertEquals(2, sockets())
        vm.liveState.awaitFirst { it is WsState.Open }
        vm.disconnectLivePhotos()
    }

    @Test fun `an upcoming event loads no gallery and opens no socket`() {
        val vm = RunnerGalleryViewModel(application)
        vm.selectEvent(event(LocalDate.now().plusDays(7)))
        assertEquals(PhotosSearchState.Idle, vm.searchState.value)
        vm.connectLivePhotos()
        Thread.sleep(200)
        assertEquals(0, photoPages())
        assertEquals(0, sockets())
        assertEquals(WsState.Idle, vm.liveState.value)
    }
}
