package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.runComposeUiTest
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.PhotographerLibraryPhotoDto
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
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
class PhotographerPurchasePreviewTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()
    private lateinit var application: Application
    private lateinit var vm: PhotographerPurchasesViewModel
    private var purging = false
    private var refundWithoutTimestamp = false

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val order = request.requestUrl?.queryParameter("orderId") ?: "a1b2c3d4-order"
                val buyer = if (order.startsWith("b")) "Bea T." else "Aira S."
                val refund = if (refundWithoutTimestamp) """{"refundStatus":"refunded","refundedPhp":50.0,"refundedAt":null,"transactions":[{"id":"refund-a","isRefund":true,"refundOf":"sale-a","amountKept":-25.5,"discountPhp":0.0,"paidAt":"2026-10-02T00:00:00Z","recordedAt":"2026-10-02T00:00:00Z"}]}""" else """{"refundStatus":"none","refundedPhp":0.0,"refundedAt":null,"transactions":[]}"""
                val receipt = """{"orderId":"$order","photoId":"photo-a","buyer":"$buyer","purchasedAt":"2026-10-01T00:00:00Z","orderCreatedAt":"2026-10-01T00:00:00Z","paymentMethod":"gcash","pricePhp":50.0,"discountPhp":0.0,"paidPhp":50.0,"kind":"paid","refundPending":false,"netKeptPhp":25.5,${refund.removeSurrounding("{", "}")}}"""
                val photo = """{"id":"photo-a","bib":"123","status":"ready","salesCount":2,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":null,"deletedAt":null,"deletionStartedAt":${if (purging) "\"2026-10-02T00:00:00Z\"" else "null"}}"""
                return ok("""{"photo":$photo,"event":{"id":"event-a","name":"Cebu Run","date":"2026-10-01","deletedAt":null},"items":[$receipt],"total":1,"offset":0,"limit":50,"selectedPurchase":$receipt}""")
            }
        }
        vm = PhotographerPurchasesViewModel(application)
    }

    @After fun tearDown() {
        vm.closePreview()
        SessionManager.getInstance(application).clearSession()
    }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `same photo new order replaces buyer and reference while unavailable image keeps details`() {
        purging = true
        vm.openPurchase("photo-a", "a1b2c3d4-order")
        vm.previewState.awaitFirst { it.selectedPurchase?.buyer == "Aira S." }
        var closed = 0
        var openedEvent: String? = null
        runComposeUiTest {
            setContent {
                PhotographerPurchasePreview(vm, onClose = { closed++ }, onOpenEvent = { openedEvent = it })
            }
            onNodeWithText("Aira S.").assertIsDisplayed()
            onNodeWithText("A1B2C3D4").assertIsDisplayed()
            onNodeWithText("Paid for this photo").assertIsDisplayed()
            onNodeWithText("Download ↓").assertDoesNotExist()
            runOnIdle { vm.openPurchase("photo-a", "b1b2c3d4-order") }
            waitUntil(timeoutMillis = 5_000) { vm.previewState.value.selectedPurchase?.buyer == "Bea T." }
            onNodeWithText("Bea T.").assertIsDisplayed()
            onNodeWithText("B1B2C3D4").assertIsDisplayed()
            onNodeWithText("Aira S.").assertDoesNotExist()
            onNodeWithText("Cebu Run").performClick()
            runOnIdle { assertEquals("event-a", openedEvent) }
        }
    }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `resolved owner photo keeps download when its thumbnail is missing`() {
        vm.openPurchase("photo-a", "a1b2c3d4-order")
        vm.previewState.awaitFirst { it.selectedPurchase != null }
        runComposeUiTest {
            setContent { PhotographerPurchasePreview(vm, onClose = {}, onOpenEvent = {}) }
            onNodeWithText("Aira S.").assertIsDisplayed()
            onNodeWithText("Download ↓").assertIsDisplayed()
        }
    }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `refund without completion timestamp shows the recorded Manila time`() {
        refundWithoutTimestamp = true
        vm.openPurchase("photo-a", "a1b2c3d4-order")
        vm.previewState.awaitFirst { it.selectedPurchase?.refundStatus == "refunded" }
        runComposeUiTest {
            setContent { PhotographerPurchasePreview(vm, onClose = {}, onOpenEvent = {}) }
            onNodeWithText("Refund recorded").assertExists()
            onNodeWithText("Oct 2, 2026 · 8:00 AM").assertExists()
            onNodeWithText("Refund completed").assertDoesNotExist()
        }
    }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `switching photos during download keeps the new photo downloadable and ignores stale completion`() {
        val aStarted = CountDownLatch(1)
        val releaseA = CountDownLatch(1)
        val aResponded = CountDownLatch(1)
        val bStarted = CountDownLatch(1)
        val releaseB = CountDownLatch(1)
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val id = if (request.path.orEmpty().contains("photo-b")) "photo-b" else "photo-a"
                return when {
                    request.path.orEmpty().contains("/download") -> {
                        if (id == "photo-a") {
                            aStarted.countDown()
                            releaseA.await(5, TimeUnit.SECONDS)
                            aResponded.countDown()
                        } else {
                            bStarted.countDown()
                            releaseB.await(5, TimeUnit.SECONDS)
                        }
                        MockResponse().setResponseCode(503)
                    }
                    request.path.orEmpty().contains("/purchases") -> {
                        val photo = """{"id":"$id","bib":null,"status":"ready","salesCount":1,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":null,"deletedAt":null,"deletionStartedAt":null}"""
                        val receipt = """{"orderId":"a1b2c3d4-order","photoId":"$id","buyer":"Aira S.","purchasedAt":"2026-10-01T00:00:00Z","orderCreatedAt":"2026-10-01T00:00:00Z","paymentMethod":"gcash","pricePhp":50.0,"discountPhp":0.0,"paidPhp":50.0,"kind":"paid","refundStatus":"none","refundPending":false,"refundedPhp":0.0,"refundedAt":null,"netKeptPhp":25.5,"transactions":[]}"""
                        ok("""{"photo":$photo,"event":{"id":"event-a","name":"Cebu Run","date":"2026-10-01","deletedAt":null},"items":[$receipt],"total":1,"offset":0,"limit":50,"selectedPurchase":$receipt}""")
                    }
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }
        val photos = listOf("photo-a", "photo-b").map { id ->
            PhotographerLibraryPhotoDto(id, null, "ready", 1, "2026-10-01T00:00:00Z", 0, "normal")
        }
        vm.openPurchase("photo-a")
        vm.previewState.awaitFirst { it.selectedPurchase?.photoId == "photo-a" }
        try {
            runComposeUiTest {
                setContent { PhotographerPurchasePreview(vm, photos = photos, onClose = {}, onOpenEvent = {}) }
                onNodeWithText("Download ↓").performClick()
                assertTrue(aStarted.await(5, TimeUnit.SECONDS))
                onNodeWithContentDescription("Next").performClick()
                waitUntil(timeoutMillis = 5_000) { vm.previewState.value.selectedPurchase?.photoId == "photo-b" }
                waitForIdle()
                onNodeWithText("Download ↓").assertExists()
                onNodeWithText("Download ↓").performClick()
                assertTrue(bStarted.await(5, TimeUnit.SECONDS))
                releaseA.countDown()
                assertTrue(aResponded.await(5, TimeUnit.SECONDS))
                waitForIdle()
                onNodeWithText("Download ↓").assertDoesNotExist()
                onNodeWithText("Download unavailable. Try again.").assertDoesNotExist()
            }
        } finally {
            releaseA.countDown()
            releaseB.countDown()
        }
    }
}
