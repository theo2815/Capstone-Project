package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.test.runComposeUiTest
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.PhotographerEventDetailDto
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.CopyOnWriteArrayList

@RunWith(RobolectricTestRunner::class)
class PhotographerEventShareScreenTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()
    private lateinit var application: Application
    private lateinit var dashboard: PhotographerDashboardViewModel
    private lateinit var purchases: PhotographerPurchasesViewModel
    private val paths = CopyOnWriteArrayList<String>()
    private var showPurchasedPhoto = false

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                paths += request.path.orEmpty()
                return if (request.path.orEmpty().contains("purchased-photos"))
                    if (showPurchasedPhoto) ok("""{"items":[{"photo":{"id":"photo-a","bib":"123","status":"ready","salesCount":1,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":null},"purchaseCount":1,"lastPurchasedAt":"2026-10-01T00:00:00Z"}],"total":1,"offset":0,"limit":50}""")
                    else ok("""{"items":[],"total":0,"offset":0,"limit":50}""")
                else if (request.path.orEmpty().contains("/photos/photo-a/purchases")) {
                    val offset = request.requestUrl!!.queryParameter("offset")!!.toInt()
                    fun receipt(n: Int) = """{"orderId":"order-$n","photoId":"photo-a","buyer":"Buyer $n","purchasedAt":"2026-10-01T00:00:00Z","orderCreatedAt":"2026-10-01T00:00:00Z","paymentMethod":"gcash","pricePhp":50,"discountPhp":0,"paidPhp":50,"kind":"paid","refundStatus":"none","refundPending":false,"refundedPhp":0,"refundedAt":null,"netKeptPhp":25.5,"transactions":[]}"""
                    val items = (if (offset == 0) 1..50 else 51..51).joinToString(",") { receipt(it) }
                    ok("""{"photo":{"id":"photo-a","bib":"123","status":"ready","salesCount":51,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":null},"event":{"id":"event-a","name":"Cebu Run","date":"2026-10-01","deletedAt":null},"items":[$items],"total":51,"offset":$offset,"limit":50,"selectedPurchase":${receipt(1)}}""")
                }
                else if (request.path.orEmpty().contains("/photos?"))
                    ok("""{"items":[{"id":"photo-a","bib":null,"status":"ready","salesCount":1,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":null}],"total":1,"offset":0,"limit":50}""")
                else MockResponse().setResponseCode(404)
            }
        }
        dashboard = PhotographerDashboardViewModel(application)
        purchases = PhotographerPurchasesViewModel(application)
    }

    @After fun tearDown() { SessionManager.getInstance(application).clearSession() }

    private fun event(removed: Boolean) = PhotographerEventDetailDto(
        id = "event-a", slug = "cebu", name = "Cebu Run", date = "2026-10-01", location = "Cebu",
        state = "past", photoCount = 1, salesCount = 1, revenueKept = 25.0,
        deletedAt = if (removed) "2026-10-02T00:00:00Z" else null,
    )

    @OptIn(ExperimentalTestApi::class)
    @Test fun `loaded receipt 51 retains both pages without a request online and offline`() {
        showPurchasedPhoto = true
        runComposeUiTest {
            setContent {
                PhotographerEventShareScreen(
                    event = event(false), viewModel = dashboard, purchaseViewModel = purchases,
                    initialTab = "purchased", onBack = {},
                )
            }
            waitUntil(timeoutMillis = 5_000) { purchases.purchasedPhotosState.value.items.size == 1 }
            onNodeWithText("1 purchase").performSemanticsAction(SemanticsActions.OnClick)
            waitForIdle()
            waitUntil(timeoutMillis = 5_000) { purchases.previewState.value.items.size == 50 }
            onNodeWithText("Load more purchases").performSemanticsAction(SemanticsActions.OnClick)
            waitForIdle()
            waitUntil(timeoutMillis = 5_000) { purchases.previewState.value.items.size == 51 }
            for (offline in listOf(false, true)) {
                if (offline) backend.server.shutdown()
                onNodeWithText("Buyer 51 · ORDER-51").performSemanticsAction(SemanticsActions.OnClick)
                waitForIdle()
                runOnIdle {
                    val state = purchases.previewState.value
                    assertEquals("order-51", state.selectedPurchase?.orderId)
                    assertEquals(51, state.items.size)
                    assertEquals(100, state.nextOffset)
                    assertEquals(2, paths.count { it.contains("/photos/photo-a/purchases") })
                }
                if (!offline) onNodeWithText("Buyer 1 · ORDER-1").performSemanticsAction(SemanticsActions.OnClick)
            }
        }
    }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `removed event forces purchased and never calls active or deleted library`() {
        runComposeUiTest {
            setContent {
                PhotographerEventShareScreen(
                    event = event(true), viewModel = dashboard, purchaseViewModel = purchases,
                    initialTab = "active", onBack = {},
                )
            }
            waitUntil(timeoutMillis = 5_000) { paths.any { it.contains("purchased-photos") } }
            onNodeWithText("Removed event", ignoreCase = true).assertExists()
            onNodeWithText("No completed purchases yet.").assertExists()
            onNodeWithText("Copy link").assertDoesNotExist()
            onNodeWithText("Select").assertDoesNotExist()
            runOnIdle {
                assertFalse(paths.any { it.contains("/photos?deleted=") || it.contains("/metadata-visibility") })
                assertTrue(paths.any { it.contains("purchased-photos") })
            }
        }
    }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `purchased tab never offers active-library selection`() {
        dashboard.fetchSharePhotos("event-a")
        dashboard.sharePhotosState.awaitFirst { it is SharePhotosState.Success }
        runComposeUiTest {
            setContent {
                PhotographerEventShareScreen(
                    event = event(false), viewModel = dashboard, purchaseViewModel = purchases,
                    initialTab = "purchased", onBack = {},
                )
            }
            onNodeWithText("Purchased").assertExists()
            onNodeWithText("Select").assertDoesNotExist()
        }
    }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `purchased grid shows one entry per photo with purchase count`() {
        showPurchasedPhoto = true
        runComposeUiTest {
            setContent {
                PhotographerEventShareScreen(
                    event = event(false), viewModel = dashboard, purchaseViewModel = purchases,
                    initialTab = "purchased", onBack = {},
                )
            }
            waitUntil(timeoutMillis = 5_000) { purchases.purchasedPhotosState.value.items.size == 1 }
            waitForIdle()
            onNodeWithText("1 purchased photos", ignoreCase = true).assertExists()
            onNodeWithText("1 purchase").assertExists()
        }
    }
}
