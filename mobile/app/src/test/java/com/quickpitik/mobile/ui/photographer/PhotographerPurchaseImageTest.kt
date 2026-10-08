package com.quickpitik.mobile.ui.photographer

import android.app.Application
import android.graphics.Bitmap
import android.os.Looper
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.runComposeUiTest
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import okio.Buffer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicInteger

@RunWith(RobolectricTestRunner::class)
class PhotographerPurchaseImageTest {
    @get:Rule val backend = MockBackend()
    private lateinit var application: Application
    private lateinit var vm: PhotographerPurchasesViewModel

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        vm = PhotographerPurchasesViewModel(application)
    }

    @After fun tearDown() {
        vm.closePreview()
        SessionManager.getInstance(application).clearSession()
    }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `real image 404 retries same URL only after bounded refresh and retains receipt`() {
        val imageCalls = AtomicInteger()
        val purchaseCalls = AtomicInteger()
        val bytes = ByteArrayOutputStream().also { stream ->
            Bitmap.createBitmap(4, 3, Bitmap.Config.ARGB_8888).compress(Bitmap.CompressFormat.PNG, 100, stream)
        }.toByteArray()
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse = when {
                request.path.orEmpty().startsWith("/photo.png") -> {
                    if (imageCalls.incrementAndGet() == 1) MockResponse().setResponseCode(404)
                    else MockResponse().setResponseCode(200).setHeader("Content-Type", "image/png").setBody(Buffer().write(bytes))
                }
                request.path.orEmpty().contains("/purchases") -> {
                    purchaseCalls.incrementAndGet()
                    val photo = """{"id":"photo-a","bib":"123","status":"ready","salesCount":2,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":"/photo.png","deletedAt":null,"deletionStartedAt":null}"""
                    val receipt = """{"orderId":"a1b2c3d4-order","photoId":"photo-a","buyer":"Aira S.","purchasedAt":"2026-10-01T00:00:00Z","orderCreatedAt":"2026-10-01T00:00:00Z","paymentMethod":"gcash","pricePhp":50.0,"discountPhp":0.0,"paidPhp":50.0,"kind":"paid","refundStatus":"none","refundPending":false,"refundedPhp":0.0,"refundedAt":null,"netKeptPhp":25.5,"transactions":[]}"""
                    ok("""{"photo":$photo,"event":{"id":"event-a","name":"Cebu Run","date":"2026-10-01","deletedAt":null},"items":[$receipt],"total":1,"offset":0,"limit":${request.requestUrl?.queryParameter("limit") ?: 50},"selectedPurchase":$receipt}""")
                }
                else -> MockResponse().setResponseCode(404)
            }
        }
        runComposeUiTest {
            setContent { PhotographerPurchasePreview(vm, onClose = {}, onOpenEvent = {}) }
            runOnIdle { vm.openPurchase("photo-a", "a1b2c3d4-order") }
            waitUntil(timeoutMillis = 10_000) { shadowOf(Looper.getMainLooper()).idle(); vm.previewState.value.selectedPurchase != null }
            waitUntil(timeoutMillis = 10_000) { shadowOf(Looper.getMainLooper()).idle(); onAllNodesWithText("Preview unavailable").fetchSemanticsNodes().isNotEmpty() }
            onNodeWithText("Aira S.").assertIsDisplayed()
            onNodeWithText("A1B2C3D4").assertIsDisplayed()
            onNodeWithText("Refresh image").performClick()
            waitUntil(timeoutMillis = 10_000) { shadowOf(Looper.getMainLooper()).idle(); vm.previewState.value.photoRefreshGeneration == 1L && imageCalls.get() == 2 }
            waitUntil(timeoutMillis = 10_000) { shadowOf(Looper.getMainLooper()).idle(); onAllNodesWithText("Preview unavailable").fetchSemanticsNodes().isEmpty() }
            onNodeWithText("Aira S.").assertIsDisplayed()
            onNodeWithText("A1B2C3D4").assertIsDisplayed()
        }
        assertEquals(2, purchaseCalls.get())
        assertEquals(2, imageCalls.get())
    }
}
