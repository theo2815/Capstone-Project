package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.hasScrollAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.performScrollToNode
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.runComposeUiTest
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
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class PhotographerEarningsScreenTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()
    private lateinit var application: Application
    private lateinit var dashboard: PhotographerDashboardViewModel
    private lateinit var earnings: StudioEarningsViewModel
    private lateinit var purchases: PhotographerPurchasesViewModel
    private var ledgerData = """{"items":[],"total":0,"offset":0,"limit":50,"monthTotals":{}}"""

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse = when {
                request.path.orEmpty().contains("earnings/per-event") -> ok("""{"items":[{"eventId":"event-zero","eventName":"Free Run","eventDate":"2026-10-01","deletedAt":null,"photoCount":1,"salesCount":0,"purchaseCount":1,"revenueKept":0.0}],"total":1,"offset":0,"limit":10}""")
                request.path.orEmpty().endsWith("/earnings") -> ok("""{"lifetimeKept":0.0,"thisWeek":0.0,"thisMonth":0.0,"payoutPending":0.0,"payoutScheduledFor":null,"weeklySeries":[],"thisWeekSold":0,"thisMonthSold":0}""")
                request.path.orEmpty().contains("payouts/balance") -> ok("""{"unpaidBalance":0.0,"clearing":0.0,"reserved":0.0,"accountCooldownUntil":null,"minimum":100.0,"hasOpenRequest":false,"openRequest":null,"primaryAccount":null}""")
                request.path.orEmpty().contains("billing/transactions") -> ok(ledgerData)
                request.path.orEmpty().contains("/photos/photo-a/purchases") -> {
                    val receipt = """{"orderId":"a1b2c3d4-order","photoId":"photo-a","buyer":"Aira S.","purchasedAt":"2026-10-01T00:00:00Z","orderCreatedAt":"2026-10-01T00:00:00Z","paymentMethod":"gcash","pricePhp":50.0,"discountPhp":0.0,"paidPhp":50.0,"kind":"paid","refundStatus":"none","refundPending":false,"refundedPhp":0.0,"refundedAt":null,"netKeptPhp":25.5,"transactions":[{"id":"tx-a","isRefund":false,"refundOf":null,"amountKept":25.5,"discountPhp":0.0,"paidAt":"2026-10-01T00:00:00Z","recordedAt":"2026-10-01T00:00:00Z"}]}"""
                    ok("""{"photo":{"id":"photo-a","bib":"123","status":"ready","salesCount":1,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":null,"deletedAt":null,"deletionStartedAt":null},"event":{"id":"event-a","name":"Transaction Run","date":"2026-10-01","deletedAt":null},"items":[$receipt],"total":1,"offset":0,"limit":50,"selectedPurchase":$receipt}""")
                }
                else -> MockResponse().setResponseCode(404)
            }
        }
        dashboard = PhotographerDashboardViewModel(application)
        earnings = StudioEarningsViewModel(application)
        purchases = PhotographerPurchasesViewModel(application)
    }

    @After fun tearDown() { SessionManager.getInstance(application).clearSession() }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `zero charge event row opens purchased history`() {
        earnings.eventEarningsState.awaitFirst { it.items.size == 1 }
        earnings.earningsUiState.awaitFirst { it is EarningsUiState.Success }
        var opened: String? = null
        runComposeUiTest {
            setContent { PhotographerEarningsScreen(dashboard, earnings, purchases, onOpenEvent = { opened = it }) }
            onNode(hasScrollAction()).performScrollToNode(hasText("Free Run"))
            onNodeWithText("Free Run").assertExists().performClick()
            runOnIdle { assertEquals("event-zero", opened) }
        }
    }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `transaction preview hides and restores the sheet then event navigation closes both`() {
        ledgerData = """{"items":[{"id":"tx-a","paidAt":"2026-10-01T00:00:00Z","eventId":"event-a","eventName":"Transaction Run","eventSlug":"run","photoId":"photo-a","orderId":"a1b2c3d4-order","buyer":"Aira S.","amountKept":25.5,"isRefund":false,"refundOf":null,"recordedAt":"2026-10-01T00:00:00Z","thumbnailUrl":null}],"total":1,"offset":0,"limit":50,"monthTotals":{"2026-10":25.5}}"""
        earnings.fetchEarningsAndTransactions()
        earnings.transactionsState.awaitFirst { it.items.size == 1 }
        earnings.earningsUiState.awaitFirst { it is EarningsUiState.Success }
        var opened: String? = null
        runComposeUiTest {
            setContent { PhotographerEarningsScreen(dashboard, earnings, purchases, onOpenEvent = { opened = it }) }
            onNode(hasScrollAction()).performScrollToNode(hasText("View transactions (1)", substring = true))
            onNodeWithText("View transactions (1)", substring = true).performClick()
            onNodeWithContentDescription("Open purchase A1B2C3D4 by Aira S.").performSemanticsAction(SemanticsActions.OnClick)
            waitForIdle()
            waitUntil(timeoutMillis = 5_000) { purchases.previewState.value.selectedPurchase != null }
            onNodeWithText("Purchase details", ignoreCase = true).assertExists()
            onNodeWithText("Recent sales history", ignoreCase = true).assertDoesNotExist()
            onNodeWithContentDescription("Close preview").performClick()
            onNodeWithText("Recent sales history", ignoreCase = true).assertExists()
            waitForIdle()
            onNodeWithContentDescription("Open purchase A1B2C3D4 by Aira S.").performSemanticsAction(SemanticsActions.OnClick)
            waitForIdle()
            waitUntil(timeoutMillis = 5_000) { purchases.previewState.value.selectedPurchase != null }
            onNodeWithText("Transaction Run", useUnmergedTree = true).performScrollTo()
            onNodeWithText("Transaction Run", useUnmergedTree = true).performClick()
            runOnIdle { assertEquals("event-a", opened) }
            onNodeWithText("Recent sales history", ignoreCase = true).assertDoesNotExist()
            onNodeWithText("Purchase details", ignoreCase = true).assertDoesNotExist()
        }
    }
}
