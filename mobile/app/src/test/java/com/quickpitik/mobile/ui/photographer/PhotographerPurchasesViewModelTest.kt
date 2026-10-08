package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.ViewModelStore
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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

@RunWith(RobolectricTestRunner::class)
class PhotographerPurchasesViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()
    private lateinit var application: Application
    private val viewModels = ViewModelStore()
    private val seenPaths = CopyOnWriteArrayList<String>()
    private var respond: (RecordedRequest) -> MockResponse = { MockResponse().setResponseCode(404) }

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                seenPaths += request.path.orEmpty()
                return respond(request)
            }
        }
    }

    @After fun tearDown() {
        viewModels.clear()
        SessionManager.getInstance(application).clearSession()
    }

    private fun model() = PhotographerPurchasesViewModel(application).also { viewModels.put("purchases", it) }
    private fun event(id: String) = """{"id":"$id","slug":"run","name":"Run","date":"2026-10-01","location":"Cebu","state":"PAST","photoCount":1,"salesCount":0,"revenueKept":0}"""
    private fun photo(id: String, thumbnail: String? = null) = """{"id":"$id","bib":null,"status":"ready","salesCount":1,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":${thumbnail?.let { "\"$it\"" } ?: "null"},"deletedAt":null,"deletionStartedAt":null}"""
    private fun receipt(order: String, photo: String) = """{"orderId":"$order","photoId":"$photo","buyer":"Aira S.","purchasedAt":"2026-10-01T00:00:00Z","orderCreatedAt":"2026-10-01T00:00:00Z","paymentMethod":"gcash","pricePhp":0,"discountPhp":0,"paidPhp":0,"kind":"free","refundStatus":"none","refundPending":false,"refundedPhp":null,"refundedAt":null,"netKeptPhp":0,"transactions":[]}"""
    private fun purchases(photoId: String, orders: List<String>, total: Int, offset: Int = 0, selected: String? = null, thumbnail: String? = null): String =
        """{"photo":${photo(photoId, thumbnail)},"event":{"id":"event-a","name":"Run","date":"2026-10-01","deletedAt":null},"items":[${orders.joinToString(",") { receipt(it, photoId) }}],"total":$total,"offset":$offset,"limit":50,"selectedPurchase":${selected?.let { receipt(it, photoId) } ?: "null"}}"""
    private fun gallery(items: List<String>, total: Int, offset: Int = 0) =
        """{"items":[${items.joinToString(",") { """{"photo":${photo(it)},"purchaseCount":1,"lastPurchasedAt":"2026-10-01T00:00:00Z"}""" }}],"total":$total,"offset":$offset,"limit":50}"""

    @Test fun `opens an event absent from cached events`() {
        respond = { request -> when {
            request.path!!.startsWith("/api/v1/me/photographer/events/event-a?") -> ok("""{"id":"event-a","slug":"run","name":"Run","date":"2026-10-01","location":"Cebu","state":"PAST","photoCount":1,"salesCount":0,"revenueKept":0,"deletedAt":"2026-10-02T01:02:03Z"}""")
            request.path!!.contains("purchased-photos") -> ok(gallery(listOf("photo-a"), 1))
            else -> MockResponse().setResponseCode(404)
        } }
        val vm = model()
        vm.loadEvent("event-a")
        vm.eventState.awaitFirst { it.event?.id == "event-a" }
        assertEquals("2026-10-02T01:02:03Z", vm.eventState.value.event?.deletedAt)
        assertTrue(seenPaths.any { it.contains("includeRemoved=true") })
        vm.loadPurchasedPhotos("event-a")
        vm.purchasedPhotosState.awaitFirst { it.items.size == 1 }
        assertEquals("photo-a", vm.purchasedPhotosState.value.items.single().photo.id)
    }

    @Test fun `older event detail without removed timestamp stays readable`() {
        respond = { request ->
            if (request.path!!.contains("/events/event-a?")) ok(event("event-a"))
            else MockResponse().setResponseCode(404)
        }
        val vm = model()
        vm.loadEvent("event-a")
        vm.eventState.awaitFirst { it.event?.id == "event-a" }
        assertNull(vm.eventState.value.event?.deletedAt)
    }

    @Test fun `initial gallery failure requires explicit retry before paging`() {
        var calls = 0
        respond = { request ->
            if (request.path!!.contains("purchased-photos")) {
                calls++
                if (calls == 1) MockResponse().setResponseCode(429) else ok(gallery(listOf("photo-a"), 1))
            } else MockResponse().setResponseCode(404)
        }
        val vm = model()
        vm.loadPurchasedPhotos("event-a")
        vm.purchasedPhotosState.awaitFirst { it.errorStatus == 429 }
        assertFalse(vm.purchasedPhotosState.value.hasMore)
        vm.loadMorePurchasedPhotos()
        assertEquals(1, calls)
        vm.loadPurchasedPhotos("event-a", refresh = true)
        vm.purchasedPhotosState.awaitFirst { it.items.size == 1 }
        assertEquals("photo-a", vm.purchasedPhotosState.value.items.single().photo.id)
        assertNull(vm.purchasedPhotosState.value.error)
        assertNull(vm.purchasedPhotosState.value.errorStatus)
    }

    @Test fun `initial preview failure requires explicit retry before paging`() {
        var calls = 0
        respond = { request ->
            if (request.path!!.contains("/purchases")) {
                calls++
                if (calls == 1) MockResponse().setResponseCode(404)
                else ok(purchases("photo-a", listOf("order-a"), 1, selected = "order-a"))
            } else MockResponse().setResponseCode(404)
        }
        val vm = model()
        vm.openPurchase("photo-a", "order-a")
        vm.previewState.awaitFirst { it.errorStatus == 404 }
        assertFalse(vm.previewState.value.hasMore)
        vm.loadMorePurchases()
        assertEquals(1, calls)
        vm.retryPreview()
        vm.previewState.awaitFirst { it.selectedPurchase?.orderId == "order-a" }
        assertEquals("photo-a", vm.previewState.value.photo?.id)
        assertEquals("event-a", vm.previewState.value.event?.id)
        assertEquals(1, vm.previewState.value.items.size)
        assertNull(vm.previewState.value.error)
        assertNull(vm.previewState.value.errorStatus)
    }

    @Test fun `pins a receipt outside the first page and ignores older selection response`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        respond = { request ->
            val order = request.requestUrl?.queryParameter("orderId")
            when {
                request.path!!.contains("/photos/photo-a/purchases") && order == "order-a" -> {
                    started.countDown()
                    release.await(5, TimeUnit.SECONDS)
                    ok(purchases("photo-a", listOf("first"), 2, selected = "order-a"))
                }
                request.path!!.contains("/photos/photo-a/purchases") -> ok(purchases("photo-a", listOf("first"), 2, selected = "order-b"))
                else -> MockResponse().setResponseCode(404)
            }
        }
        val vm = model()
        vm.openPurchase("photo-a", "order-a")
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try {
            vm.openPurchase("photo-a", "order-b")
            vm.previewState.awaitFirst { it.selectedPurchase?.orderId == "order-b" }
        } finally { release.countDown() }
        Thread.sleep(200)
        assertEquals("order-b", vm.previewState.value.selectedPurchase?.orderId)
        assertEquals(1, vm.previewState.value.items.size)
    }

    @Test fun `older event detail cannot replace the newly opened event`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        respond = { request -> when {
            request.path!!.contains("/events/event-a?") -> {
                started.countDown()
                release.await(5, TimeUnit.SECONDS)
                ok(event("event-a"))
            }
            request.path!!.contains("/events/event-b?") -> ok(event("event-b"))
            else -> MockResponse().setResponseCode(404)
        } }
        val vm = model()
        vm.loadEvent("event-a")
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try {
            vm.loadEvent("event-b")
            vm.eventState.awaitFirst { it.event?.id == "event-b" }
        } finally { release.countDown() }
        Thread.sleep(200)
        assertEquals("event-b", vm.eventState.value.event?.id)
    }

    @Test fun `older gallery cannot replace another event's purchased photos`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        respond = { request -> when {
            request.path!!.contains("/events/event-a/purchased-photos") -> {
                started.countDown()
                release.await(5, TimeUnit.SECONDS)
                ok(gallery(listOf("photo-a"), 1))
            }
            request.path!!.contains("/events/event-b/purchased-photos") -> ok(gallery(listOf("photo-b"), 1))
            else -> MockResponse().setResponseCode(404)
        } }
        val vm = model()
        vm.loadPurchasedPhotos("event-a")
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try {
            vm.loadPurchasedPhotos("event-b")
            vm.purchasedPhotosState.awaitFirst { it.items.singleOrNull()?.photo?.id == "photo-b" }
        } finally { release.countDown() }
        Thread.sleep(200)
        assertEquals("event-b", vm.purchasedPhotosState.value.eventId)
        assertEquals("photo-b", vm.purchasedPhotosState.value.items.single().photo.id)
    }

    @Test fun `older photo purchase response cannot replace another photo's preview`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        respond = { request -> when {
            request.path!!.contains("/photos/photo-a/purchases") -> {
                started.countDown()
                release.await(5, TimeUnit.SECONDS)
                ok(purchases("photo-a", listOf("order-a"), 1))
            }
            request.path!!.contains("/photos/photo-b/purchases") -> ok(purchases("photo-b", listOf("order-b"), 1))
            else -> MockResponse().setResponseCode(404)
        } }
        val vm = model()
        vm.openPurchase("photo-a")
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try {
            vm.openPurchase("photo-b")
            vm.previewState.awaitFirst { it.selectedPurchase?.orderId == "order-b" }
        } finally { release.countDown() }
        Thread.sleep(200)
        assertEquals("photo-b", vm.previewState.value.photoId)
        assertEquals("order-b", vm.previewState.value.selectedPurchase?.orderId)
    }

    @Test fun `gallery advances raw offsets and stops on empty page with stale total`() {
        respond = { request -> when {
            request.path!!.contains("purchased-photos") -> {
                val offset = request.requestUrl?.queryParameter("offset")?.toInt() ?: 0
                ok(gallery(if (offset == 0) listOf("photo-a", "photo-b") else if (offset == 50) listOf("photo-b") else emptyList(), 150, offset))
            }
            else -> MockResponse().setResponseCode(404)
        } }
        val vm = model()
        vm.loadPurchasedPhotos("event-a")
        vm.purchasedPhotosState.awaitFirst { it.items.size == 2 }
        vm.loadMorePurchasedPhotos()
        vm.purchasedPhotosState.awaitFirst { it.nextOffset == 100 }
        assertEquals(2, vm.purchasedPhotosState.value.items.size)
        vm.loadMorePurchasedPhotos()
        vm.purchasedPhotosState.awaitFirst { !it.hasMore }
        assertEquals(2, vm.purchasedPhotosState.value.items.size)
        assertTrue(seenPaths.any { it.contains("offset=100") })
    }

    @Test fun `preview exposes 404 and 429 for manual retry`() {
        var status = 404
        respond = { request -> if (request.path!!.contains("/purchases")) MockResponse().setResponseCode(status) else MockResponse().setResponseCode(404) }
        val vm = model()
        vm.openPurchase("photo-a")
        vm.previewState.awaitFirst { it.error != null }
        assertEquals(404, vm.previewState.value.errorStatus)
        status = 429
        vm.retryPreview()
        vm.previewState.awaitFirst { it.errorStatus == 429 }
        assertEquals(2, seenPaths.size)
    }

    @Test fun `closing preview discards an in-flight purchase response`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        respond = {
            started.countDown()
            release.await(5, TimeUnit.SECONDS)
            ok(purchases("photo-a", listOf("order-a"), 1))
        }
        val vm = model()
        vm.openPurchase("photo-a")
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try { vm.closePreview() } finally { release.countDown() }
        Thread.sleep(200)
        assertNull(vm.previewState.value.photoId)
        assertNull(vm.previewState.value.selectedPurchase)
    }

    @Test fun `selecting a loaded receipt survives an append response`() {
        respond = { request ->
            val offset = request.requestUrl?.queryParameter("offset")?.toInt() ?: 0
            if (offset == 0) ok(purchases("photo-a", listOf("order-a", "order-b"), 100, selected = "order-a"))
            else ok(purchases("photo-a", listOf("order-c"), 100, offset)).setBodyDelay(300, TimeUnit.MILLISECONDS)
        }
        val vm = model()
        vm.openPurchase("photo-a")
        vm.previewState.awaitFirst { it.items.size == 2 }
        vm.loadMorePurchases()
        vm.selectPurchase("order-b")
        vm.previewState.awaitFirst { it.nextOffset == 100 && !it.isLoadingMore }
        assertEquals("order-b", vm.previewState.value.selectedPurchase?.orderId)
    }

    @Test fun `a missing pinned receipt never selects another buyer`() {
        respond = { ok(purchases("photo-a", listOf("order-a"), 2)) }
        val vm = model()
        vm.openPurchase("photo-a", "order-b")
        vm.previewState.awaitFirst { !it.isLoading && it.items.isNotEmpty() }
        assertNull(vm.previewState.value.selectedPurchase)
    }

    @Test fun `gallery append retry clears its prior rate limit error`() {
        var appendCalls = 0
        respond = { request ->
            val offset = request.requestUrl?.queryParameter("offset")?.toInt() ?: 0
            if (offset == 0) ok(gallery(listOf("photo-a"), 100))
            else if (++appendCalls == 1) MockResponse().setResponseCode(429)
            else ok(gallery(listOf("photo-b"), 100, offset))
        }
        val vm = model()
        vm.loadPurchasedPhotos("event-a")
        vm.purchasedPhotosState.awaitFirst { it.items.size == 1 }
        vm.loadMorePurchasedPhotos()
        vm.purchasedPhotosState.awaitFirst { it.appendError != null }
        assertEquals(1, vm.purchasedPhotosState.value.items.size)
        assertEquals(429, vm.purchasedPhotosState.value.errorStatus)
        vm.loadMorePurchasedPhotos()
        vm.purchasedPhotosState.awaitFirst { it.items.size == 2 }
        assertNull(vm.purchasedPhotosState.value.appendError)
        assertNull(vm.purchasedPhotosState.value.errorStatus)
    }

    @Test fun `receipt append retry clears its prior rate limit error`() {
        var appendCalls = 0
        respond = { request ->
            val offset = request.requestUrl?.queryParameter("offset")?.toInt() ?: 0
            if (offset == 0) ok(purchases("photo-a", listOf("order-a"), 100, selected = "order-a"))
            else if (++appendCalls == 1) MockResponse().setResponseCode(429)
            else ok(purchases("photo-a", listOf("order-b"), 100, offset))
        }
        val vm = model()
        vm.openPurchase("photo-a")
        vm.previewState.awaitFirst { it.items.size == 1 }
        vm.loadMorePurchases()
        vm.previewState.awaitFirst { it.appendError != null }
        assertEquals(1, vm.previewState.value.items.size)
        assertEquals(429, vm.previewState.value.errorStatus)
        vm.loadMorePurchases()
        vm.previewState.awaitFirst { it.items.size == 2 }
        assertNull(vm.previewState.value.appendError)
        assertNull(vm.previewState.value.errorStatus)
    }

    @Test fun `old account responses cannot restore purchase state`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        respond = { request -> if (request.path!!.contains("/purchases")) {
            started.countDown()
            release.await(5, TimeUnit.SECONDS)
            ok(purchases("photo-a", listOf("order-a"), 1))
        } else MockResponse().setResponseCode(404) }
        val vm = model()
        vm.openPurchase("photo-a")
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try { signInAs("RUNNER") } finally { release.countDown() }
        vm.previewState.awaitFirst { it.photoId == null }
        Thread.sleep(200)
        assertNull(vm.previewState.value.photoId)
        assertNull(vm.previewState.value.selectedPurchase)
        assertFalse(vm.previewState.value.isLoading)
    }

    @Test fun `logout discards an in-flight purchase response`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        respond = {
            started.countDown()
            release.await(5, TimeUnit.SECONDS)
            ok(purchases("photo-a", listOf("order-a"), 1))
        }
        val vm = model()
        vm.openPurchase("photo-a")
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try { SessionManager.getInstance(application).clearSession() } finally { release.countDown() }
        Thread.sleep(200)
        assertNull(vm.previewState.value.photoId)
        assertNull(vm.previewState.value.selectedPurchase)
        assertFalse(vm.previewState.value.isLoading)
    }

    @Test fun `media refresh is bounded and preserves pinned purchase history and selection`() {
        var refreshed = false
        val firstPage = (1..50).map { "order-$it" }
        respond = { request ->
            val offset = request.requestUrl?.queryParameter("offset")?.toInt() ?: 0
            val limit = request.requestUrl?.queryParameter("limit")?.toInt() ?: 50
            if (offset == 50) ok(purchases("photo-a", listOf("order-51"), 51, offset))
            else if (limit == 1) {
                refreshed = true
                ok(purchases("photo-a", listOf("order-1"), 51, selected = "order-51", thumbnail = "/same.jpg"))
            } else ok(purchases("photo-a", firstPage, 51, selected = "order-51", thumbnail = "/same.jpg"))
        }
        val vm = model()
        vm.openPurchase("photo-a", "order-51")
        vm.previewState.awaitFirst { it.selectedPurchase?.orderId == "order-51" }
        vm.loadMorePurchases()
        vm.previewState.awaitFirst { it.items.size == 51 }
        val before = vm.previewState.value
        vm.refreshPhoto()
        vm.previewState.awaitFirst { it.photoRefreshGeneration == 1L }
        val after = vm.previewState.value
        assertTrue(refreshed)
        assertTrue(seenPaths.last().contains("offset=0"))
        assertTrue(seenPaths.last().contains("limit=1"))
        assertTrue(seenPaths.last().contains("orderId=order-51"))
        assertEquals(before.items, after.items)
        assertEquals(before.total, after.total)
        assertEquals(before.nextOffset, after.nextOffset)
        assertEquals(before.selectedPurchase, after.selectedPurchase)
        assertEquals(before.photo?.thumbnailUrl, after.photo?.thumbnailUrl)
    }

    @Test fun `failed media refresh keeps receipts and duplicate taps do not start another request`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        var refreshCalls = 0
        respond = { request ->
            if (request.requestUrl?.queryParameter("limit") == "1") {
                refreshCalls++
                started.countDown()
                release.await(5, TimeUnit.SECONDS)
                MockResponse().setResponseCode(503)
            } else ok(purchases("photo-a", listOf("first"), 51, selected = "order-51"))
        }
        val vm = model()
        vm.openPurchase("photo-a", "order-51")
        vm.previewState.awaitFirst { it.selectedPurchase != null }
        vm.refreshPhoto()
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try { vm.refreshPhoto(); assertEquals(1, refreshCalls) } finally { release.countDown() }
        vm.previewState.awaitFirst { it.photoRefreshError != null }
        assertEquals("order-51", vm.previewState.value.selectedPurchase?.orderId)
        assertEquals(0L, vm.previewState.value.photoRefreshGeneration)
    }

    @Test fun `stale media refresh cannot restore a closed preview`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        respond = { request ->
            if (request.requestUrl?.queryParameter("limit") == "1") {
                started.countDown()
                release.await(5, TimeUnit.SECONDS)
            }
            ok(purchases("photo-a", listOf("first"), 1, selected = "first", thumbnail = "/same.jpg"))
        }
        val vm = model()
        vm.openPurchase("photo-a", "first")
        vm.previewState.awaitFirst { it.selectedPurchase != null }
        vm.refreshPhoto()
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try { vm.closePreview() } finally { release.countDown() }
        Thread.sleep(200)
        assertNull(vm.previewState.value.photoId)
        assertEquals(0L, vm.previewState.value.photoRefreshGeneration)
    }

    @Test fun `stale account media refresh cannot restore buyer context`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        respond = { request ->
            if (request.requestUrl?.queryParameter("limit") == "1") {
                started.countDown()
                release.await(5, TimeUnit.SECONDS)
            }
            ok(purchases("photo-a", listOf("first"), 1, selected = "first", thumbnail = "/same.jpg"))
        }
        val vm = model()
        vm.openPurchase("photo-a", "first")
        vm.previewState.awaitFirst { it.selectedPurchase != null }
        vm.refreshPhoto()
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try { signInAs("RUNNER") } finally { release.countDown() }
        vm.previewState.awaitFirst { it.photoId == null }
        Thread.sleep(200)
        assertNull(vm.previewState.value.selectedPurchase)
        assertEquals(0L, vm.previewState.value.photoRefreshGeneration)
    }
}
