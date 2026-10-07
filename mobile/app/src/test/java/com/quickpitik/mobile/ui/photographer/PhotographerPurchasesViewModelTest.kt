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
    private fun photo(id: String) = """{"id":"$id","bib":null,"status":"ready","salesCount":1,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":null,"deletedAt":null,"deletionStartedAt":null}"""
    private fun receipt(order: String, photo: String) = """{"orderId":"$order","photoId":"$photo","buyer":"Aira S.","purchasedAt":"2026-10-01T00:00:00Z","orderCreatedAt":"2026-10-01T00:00:00Z","paymentMethod":"gcash","pricePhp":0,"discountPhp":0,"paidPhp":0,"kind":"free","refundStatus":"none","refundPending":false,"refundedPhp":null,"refundedAt":null,"netKeptPhp":0,"transactions":[]}"""
    private fun purchases(photoId: String, orders: List<String>, total: Int, offset: Int = 0, selected: String? = null): String =
        """{"photo":${photo(photoId)},"event":{"id":"event-a","name":"Run","date":"2026-10-01","deletedAt":null},"items":[${orders.joinToString(",") { receipt(it, photoId) }}],"total":$total,"offset":$offset,"limit":50,"selectedPurchase":${selected?.let { receipt(it, photoId) } ?: "null"}}"""
    private fun gallery(items: List<String>, total: Int, offset: Int = 0) =
        """{"items":[${items.joinToString(",") { """{"photo":${photo(it)},"purchaseCount":1,"lastPurchasedAt":"2026-10-01T00:00:00Z"}""" }}],"total":$total,"offset":$offset,"limit":50}"""

    @Test fun `opens an event absent from cached events`() {
        respond = { request -> when {
            request.path!!.startsWith("/api/v1/me/photographer/events/event-a?") -> ok("""{"id":"event-a","slug":"run","name":"Run","date":"2026-10-01","location":"Cebu","state":"PAST","photoCount":1,"salesCount":0,"revenueKept":0}""")
            request.path!!.contains("purchased-photos") -> ok(gallery(listOf("photo-a"), 1))
            else -> MockResponse().setResponseCode(404)
        } }
        val vm = model()
        vm.loadEvent("event-a")
        vm.eventState.awaitFirst { it.event?.id == "event-a" }
        assertTrue(seenPaths.any { it.contains("includeRemoved=true") })
        vm.loadPurchasedPhotos("event-a")
        vm.purchasedPhotosState.awaitFirst { it.items.size == 1 }
        assertEquals("photo-a", vm.purchasedPhotosState.value.items.single().photo.id)
    }

    @Test fun `pins a receipt outside the first page and ignores older selection response`() {
        respond = { request ->
            val order = request.requestUrl?.queryParameter("orderId")
            when {
                request.path!!.contains("/photos/photo-a/purchases") && order == "order-a" -> ok(purchases("photo-a", listOf("first"), 2, selected = "order-a")).setBodyDelay(300, TimeUnit.MILLISECONDS)
                request.path!!.contains("/photos/photo-a/purchases") -> ok(purchases("photo-a", listOf("first"), 2, selected = "order-b"))
                else -> MockResponse().setResponseCode(404)
            }
        }
        val vm = model()
        vm.openPurchase("photo-a", "order-a")
        vm.openPurchase("photo-a", "order-b")
        vm.previewState.awaitFirst { it.selectedPurchase?.orderId == "order-b" }
        Thread.sleep(400)
        assertEquals("order-b", vm.previewState.value.selectedPurchase?.orderId)
        assertEquals(1, vm.previewState.value.items.size)
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

    @Test fun `preview exposes 404 and 429 for manual retry and close cancels work`() {
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
        vm.closePreview()
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
        respond = { request -> if (request.path!!.contains("/purchases")) ok(purchases("photo-a", listOf("order-a"), 1)).setBodyDelay(300, TimeUnit.MILLISECONDS) else MockResponse().setResponseCode(404) }
        val vm = model()
        vm.openPurchase("photo-a")
        signInAs("RUNNER")
        vm.previewState.awaitFirst { it.photoId == null }
        Thread.sleep(400)
        assertNull(vm.previewState.value.selectedPurchase)
        assertFalse(vm.previewState.value.isLoading)
    }
}
