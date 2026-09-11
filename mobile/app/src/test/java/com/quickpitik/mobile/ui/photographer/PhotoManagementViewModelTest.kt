package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.ViewModelStore
import androidx.test.core.app.ApplicationProvider
import com.google.gson.Gson
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.support.MainDispatcherRule
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class PhotoManagementViewModelTest {
    @get:Rule val main = MainDispatcherRule()

    private lateinit var application: Application
    private lateinit var session: SessionManager
    private val server = MockWebServer()
    private val viewModels = ViewModelStore()
    private val gson = Gson()
    @Volatile private var respond: (RecordedRequest) -> MockResponse = { MockResponse().setResponseCode(404) }

    @Before
    fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        session = SessionManager.getInstance(application)
        session.clearSession()
        session.saveSession("t", "PHOTOGRAPHER", "Test", "photographer@example.com")
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest) = respond(request)
        }
        server.start()
        RetrofitClient.setBaseUrl(server.url("/").toString())
    }

    @After
    fun tearDown() {
        viewModels.clear()
        AppDatabase.getDatabase(application).close()
        server.shutdown()
        RetrofitClient.resetBaseUrl()
        session.clearSession()
    }

    @Test
    fun mutationRefreshKeepsTheEventAvailableToTheShareRoute() = runBlocking {
        val refreshStarted = CountDownLatch(1)
        val finishRefresh = CountDownLatch(1)
        var eventReads = 0
        respond = { request ->
            when {
                request.path!!.substringBefore('?').endsWith("/photographer/events") -> {
                    if (++eventReads > 1) {
                        refreshStarted.countDown()
                        finishRefresh.await(5, TimeUnit.SECONDS)
                        MockResponse().setResponseCode(503)
                    } else ok("""{"items":[$event],"total":1,"offset":0,"limit":100}""")
                }
                request.path!!.endsWith("/trash") -> ok("""{"affected":1}""")
                request.path!!.contains("/photos?") -> photoPage(0, 0)
                else -> MockResponse().setResponseCode(404)
            }
        }
        val model = model()
        withTimeout(10_000) { model.eventsState.first { it is EventsState.Success } }
        try {
            assertEquals(1, model.trashPhotos("event-1", listOf("photo-1")).getOrThrow())
            assertTrue(refreshStarted.await(5, TimeUnit.SECONDS))
            assertEquals("event-1", (model.eventsState.value as EventsState.Success).events.single().id)
        } finally {
            finishRefresh.countDown()
        }
        // An unsuccessful refresh also leaves the cached route usable.
        model.fetchEvents().join()
        assertTrue(model.eventsState.value is EventsState.Success)
    }

    @Test
    fun trashLoadsBeyondTheFirstHundredAndRetainsPhotosForRetry() = runBlocking {
        val offsets = CopyOnWriteArrayList<Int>()
        var failSecondPage = true
        respond = { request ->
            if (request.requestUrl!!.queryParameter("deleted") == "true") {
                val offset = request.requestUrl!!.queryParameter("offset")!!.toInt()
                offsets += offset
                if (offset == 100 && failSecondPage) {
                    failSecondPage = false
                    MockResponse().setResponseCode(503)
                } else photoPage(offset, 205)
            } else MockResponse().setResponseCode(404)
        }
        val model = model()
        model.fetchTrashPhotos("event-1")
        awaitTrash(model) { it.photos.size == 100 }
        model.fetchTrashPhotos("event-1", loadMore = true)
        assertEquals(100, awaitTrash(model) { it.loadMoreError != null }.photos.size)
        model.fetchTrashPhotos("event-1", loadMore = true)
        awaitTrash(model) { it.photos.size == 200 }
        model.fetchTrashPhotos("event-1", loadMore = true)
        assertFalse(awaitTrash(model) { it.photos.size == 205 }.hasMore)
        model.fetchTrashPhotos("event-1", loadMore = true)
        assertEquals(listOf(0, 100, 100, 200), offsets.toList())
    }

    @Test
    fun permanentDeletionBatchesAndReportsTheUnfinishedIds() = runBlocking {
        val batches = CopyOnWriteArrayList<List<String>>()
        respond = { request ->
            if (request.path!!.endsWith("/permanent-delete")) {
                val ids = gson.fromJson(request.body.readUtf8(), Map::class.java)["photoIds"] as List<*>
                val batch = ids.map { it as String }
                batches += batch
                if (batches.size == 2) MockResponse().setResponseCode(503)
                else ok(gson.toJson(mapOf("deleted" to batch.drop(1), "blockedSold" to emptyList<String>(),
                    "blockedPending" to batch.take(1), "failed" to emptyList<String>())))
            } else MockResponse().setResponseCode(404)
        }
        val result = model().permanentDeletePhotos("event-1", (0 until 205).map { "photo-$it" }).getOrThrow()
        assertEquals(listOf(100, 100, 5), batches.map { it.size })
        assertEquals(103, result.deleted.size)
        assertEquals(listOf("photo-0", "photo-200"), result.blockedPending)
        assertEquals((100 until 200).map { "photo-$it" }, result.failed)
    }

    private fun model() = PhotographerDashboardViewModel(application).also { viewModels.put("photographer", it) }

    private suspend fun awaitTrash(model: PhotographerDashboardViewModel, accept: (SharePhotosState.Success) -> Boolean) =
        withTimeout(10_000) {
            model.trashPhotosState.first { it is SharePhotosState.Success && !it.loadingMore && accept(it) }
        } as SharePhotosState.Success

    private fun ok(data: String) = MockResponse().setBody("""{"success":true,"data":$data}""")

    private fun photoPage(offset: Int, total: Int): MockResponse {
        val photos = (offset until minOf(offset + 100, total)).map {
            mapOf("id" to "photo-$it", "status" to "live", "salesCount" to 0,
                "uploadedAt" to "2026-09-07T00:00:00Z", "tone" to 0, "span" to "default",
                "deletedAt" to "2026-09-08T00:00:00Z")
        }
        return ok(gson.toJson(mapOf("items" to photos, "offset" to offset, "limit" to 100, "total" to total)))
    }

    private val event = """{"id":"event-1","slug":"test","name":"Cebu Run","date":"2026-09-07","location":"Cebu","state":"past","photoCount":1,"salesCount":0,"revenueKept":0}"""
}
