package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * The heart toggle shared by discovery and the race log: a save shows at once
 * and is swapped for the server's row, a refused save or unsave puts the list
 * back, an unsave offers Undo, and a signed-out tap asks for a login.
 */
@RunWith(RobolectricTestRunner::class)
class SavedEventsViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private val requests = AtomicInteger()
    @Volatile private var save: MockResponse = ok(row("e1", "Cebu Run", SERVER_SAVED_AT))
    @Volatile private var unsave: MockResponse = ok("""{"removed":true}""")
    @Volatile private var holdSave: CountDownLatch? = null

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        SessionManager.getInstance(application).apply {
            clearSession()
            saveSession("t", "RUNNER", "Juan", "juan@x.com")
        }
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                requests.incrementAndGet()
                return when (request.method) {
                    "GET" -> ok("[${row("e1", "Cebu Run", SERVER_SAVED_AT)}]")
                    "POST" -> save.also { holdSave?.await(5, TimeUnit.SECONDS) }
                    else -> unsave
                }
            }
        }
    }

    @After fun tearDown() {
        holdSave?.countDown()
        SessionManager.getInstance(application).clearSession()
    }

    private fun row(id: String, name: String, savedAt: String) =
        """{"id":"$id","slug":"$id","name":"$name","date":"2026-08-01","state":"ACTIVE","bannerUrl":null,"location":"Cebu","savedAt":"$savedAt"}"""

    private fun event(id: String, name: String) = EventDto(id, id, name, "2026-08-01", "Cebu", null, 0, 0, "ACTIVE")

    @Test fun `a save shows at once, takes the server's row, and a refused save puts the list back`() {
        val vm = SavedEventsViewModel(application)
        val held = CountDownLatch(1).also { holdSave = it }
        vm.toggle(event("e1", "Cebu Run"))
        assertEquals(listOf("e1"), vm.savedEvents.value.map { it.id }) // before the server has answered
        held.countDown()
        vm.savedEvents.awaitFirst { it.singleOrNull()?.savedAt == SERVER_SAVED_AT }
        assertEquals("Saved Cebu Run.", vm.message.awaitFirst { it != null })

        holdSave = null
        save = MockResponse().setResponseCode(409)
            .setBody("""{"success":false,"errors":[{"code":"EVENT_NOT_FOUND","message":"That event is gone."}]}""")
        vm.toggle(event("e2", "Night Run"))
        assertEquals("That event is gone.", vm.message.awaitFirst { it == "That event is gone." })
        assertEquals(listOf("e1"), vm.savedEvents.value.map { it.id })
    }

    @Test fun `an unsave offers undo, a refused unsave restores the row, and signed out asks for a login`() {
        val vm = SavedEventsViewModel(application)
        vm.refresh()
        vm.savedEvents.awaitFirst { it.isNotEmpty() }
        assertTrue(vm.isSaved("e1"))

        vm.toggle(event("e1", "Cebu Run"))
        assertEquals("Removed Cebu Run from saved.", vm.message.awaitFirst { it != null })
        assertEquals("e1" to "Cebu Run", vm.undoCandidate.value)
        assertTrue(vm.savedEvents.value.isEmpty())

        vm.clearMessage()
        vm.undoUnsave("e1", "Cebu Run")
        assertEquals("Restored Cebu Run.", vm.message.awaitFirst { it != null })
        assertEquals(listOf("e1"), vm.savedEvents.value.map { it.id })

        vm.clearMessage()
        unsave = MockResponse().setResponseCode(500)
        vm.unsave("e1", "Cebu Run")
        assertEquals("HTTP 500", vm.message.awaitFirst { it != null })
        assertEquals(listOf("e1"), vm.savedEvents.value.map { it.id })

        SessionManager.getInstance(application).clearSession()
        val before = requests.get()
        vm.toggle(event("e2", "Night Run"))
        assertEquals("Please log in to save events.", vm.message.value)
        assertEquals(before, requests.get())
    }

    private companion object {
        const val SERVER_SAVED_AT = "2026-09-11T00:00:00Z"
    }
}
