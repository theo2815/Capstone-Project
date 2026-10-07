package com.quickpitik.mobile

import android.app.Application
import androidx.compose.ui.test.junit4.StateRestorationTester
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.hasScrollAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performScrollToNode
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.semantics.SemanticsActions
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.navigation
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import com.quickpitik.mobile.ui.photographer.PhotographerDashboardViewModel
import com.quickpitik.mobile.ui.photographer.PhotographerPurchasesViewModel
import com.quickpitik.mobile.ui.photographer.PhotographerEarningsScreen
import com.quickpitik.mobile.ui.photographer.StudioEarningsViewModel
import com.quickpitik.mobile.ui.photographer.EarningsUiState
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.rules.TestRule
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.atomic.AtomicInteger

@RunWith(RobolectricTestRunner::class)
class StudioShareRestorationTest {
    private val backend = MockBackend()
    private val compose = createComposeRule()
    @get:Rule val rules: TestRule = RuleChain.outerRule(backend).around(compose)
    private lateinit var application: Application
    private val purchaseCalls = AtomicInteger()
    private val eventCalls = AtomicInteger()

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse = when {
                request.path.orEmpty().contains("/events/event-a?includeRemoved=true") -> {
                    eventCalls.incrementAndGet()
                    ok("""{"id":"event-a","slug":"cebu","name":"Cebu Run","date":"2026-10-01","location":"Cebu","state":"past","photoCount":1,"salesCount":1,"revenueKept":25.5,"deletedAt":null}""")
                }
                request.path.orEmpty().contains("purchased-photos") -> ok("""{"items":[{"photo":{"id":"photo-a","bib":"123","status":"ready","salesCount":1,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":null},"purchaseCount":1,"lastPurchasedAt":"2026-10-01T00:00:00Z"}],"total":1,"offset":0,"limit":50}""")
                request.path.orEmpty().contains("/photos/photo-a/purchases") -> {
                    val call = purchaseCalls.incrementAndGet()
                    val buyer = if (call == 1) "Aira S." else "Bea T."
                    val signedUrl = backend.server.url("/signed-$call").toString()
                    val photo = """{"id":"photo-a","bib":"123","status":"ready","salesCount":1,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":"$signedUrl","deletedAt":null,"deletionStartedAt":null}"""
                    val receipt = """{"orderId":"a1b2c3d4-order","photoId":"photo-a","buyer":"$buyer","purchasedAt":"2026-10-01T00:00:00Z","orderCreatedAt":"2026-10-01T00:00:00Z","paymentMethod":"gcash","pricePhp":50.0,"discountPhp":0.0,"paidPhp":50.0,"kind":"paid","refundStatus":"none","refundPending":false,"refundedPhp":0.0,"refundedAt":null,"netKeptPhp":25.5,"transactions":[]}"""
                    ok("""{"photo":$photo,"event":{"id":"event-a","name":"Cebu Run","date":"2026-10-01","deletedAt":null},"items":[$receipt],"total":1,"offset":0,"limit":50,"selectedPurchase":$receipt}""")
                }
                request.path.orEmpty().contains("earnings/per-event") -> ok("""{"items":[],"total":0,"offset":0,"limit":10}""")
                request.path.orEmpty().endsWith("/earnings") -> ok("""{"lifetimeKept":25.5,"thisWeek":25.5,"thisMonth":25.5,"payoutPending":0.0,"payoutScheduledFor":null,"weeklySeries":[],"thisWeekSold":1,"thisMonthSold":1}""")
                request.path.orEmpty().contains("payouts/balance") -> ok("""{"unpaidBalance":0.0,"clearing":0.0,"reserved":0.0,"accountCooldownUntil":null,"minimum":100.0,"hasOpenRequest":false,"openRequest":null,"primaryAccount":null}""")
                request.path.orEmpty().contains("billing/transactions") -> ok("""{"items":[{"id":"tx-a","paidAt":"2026-10-01T00:00:00Z","eventId":"event-a","eventName":"Cebu Run","eventSlug":"cebu","photoId":"photo-a","orderId":"a1b2c3d4-order","buyer":"Aira S.","amountKept":25.5,"isRefund":false,"refundOf":null,"recordedAt":"2026-10-01T00:00:00Z","thumbnailUrl":null}],"total":1,"offset":0,"limit":50,"monthTotals":{"2026-10":25.5}}""")
                else -> MockResponse().setResponseCode(404)
            }
        }
    }

    @After fun tearDown() { SessionManager.getInstance(application).clearSession() }

    @Test fun `NavHost restores photo id and refetches private detail then clears it on session change`() {
        lateinit var nav: NavHostController
        lateinit var purchases: PhotographerPurchasesViewModel
        val restore = StateRestorationTester(compose)
        restore.setContent {
            nav = rememberNavController()
            NavHost(nav, startDestination = "studio") {
                navigation(startDestination = "studio/home", route = "studio") {
                    composable("studio/home") {}
                    composable(
                        "studio/share/{eventId}?tab={tab}",
                        arguments = listOf(
                            navArgument("eventId") { type = NavType.StringType },
                            navArgument("tab") { type = NavType.StringType; defaultValue = "active" },
                        ),
                    ) { entry ->
                        val graph = nav.getBackStackEntry("studio")
                        val dashboard: PhotographerDashboardViewModel = viewModel(graph)
                        purchases = viewModel(graph)
                        StudioShareDestination(
                            eventId = entry.arguments?.getString("eventId"),
                            initialTab = entry.arguments?.getString("tab") ?: "active",
                            dashboard = dashboard,
                            purchases = purchases,
                            onBack = { nav.popBackStack() },
                            onOpenEvent = {},
                        )
                    }
                }
            }
        }
        compose.runOnIdle { nav.navigate(shareRoute("event-a", "purchased")) }
        compose.waitForIdle()
        compose.waitUntil(5_000) { purchases.purchasedPhotosState.value.items.size == 1 }
        compose.waitForIdle()
        compose.onNodeWithText("1 purchase").performScrollTo().performSemanticsAction(SemanticsActions.OnClick)
        compose.waitForIdle()
        compose.waitUntil(5_000) { purchases.previewState.value.selectedPurchase?.buyer == "Aira S." }
        compose.onNodeWithText("Aira S.").assertExists()
        val firstVm = purchases
        val firstUrl = purchases.previewState.value.photo?.thumbnailUrl
        assertEquals(1, purchaseCalls.get())

        restore.emulateSavedInstanceStateRestore()
        compose.waitForIdle()
        compose.waitUntil(5_000) { purchaseCalls.get() >= 2 && purchases.previewState.value.selectedPurchase?.buyer == "Bea T." }
        compose.waitForIdle()
        compose.onNodeWithText("Bea T.").assertExists()
        compose.onNodeWithText("Aira S.").assertDoesNotExist()
        assertNotEquals(firstUrl, purchases.previewState.value.photo?.thumbnailUrl)
        assertTrue(eventCalls.get() >= 2)
        assertEquals("photo-a", purchases.previewState.value.photoId)

        SessionManager.getInstance(application).saveSession("other", "RUNNER", "Other", "other@x.com")
        compose.waitUntil(5_000) { purchases.previewState.value.photoId == null }
        compose.waitForIdle()
        compose.onNodeWithText("Bea T.").assertDoesNotExist()
        assertTrue(firstVm.previewState.value.photoId == null)
    }

    @Test fun `NavHost restores transaction ids and refetches their buyer and URL`() {
        lateinit var nav: NavHostController
        lateinit var earnings: StudioEarningsViewModel
        lateinit var purchases: PhotographerPurchasesViewModel
        val restore = StateRestorationTester(compose)
        restore.setContent {
            nav = rememberNavController()
            NavHost(nav, startDestination = "studio") {
                navigation(startDestination = "studio/home", route = "studio") {
                    composable("studio/home") {}
                    composable("studio/earnings") {
                        val graph = nav.getBackStackEntry("studio")
                        val dashboard: PhotographerDashboardViewModel = viewModel(graph)
                        earnings = viewModel(graph)
                        purchases = viewModel(graph)
                        PhotographerEarningsScreen(dashboard, earnings, purchases, onOpenEvent = {})
                    }
                }
            }
        }
        compose.runOnIdle { nav.navigate("studio/earnings") }
        compose.waitForIdle()
        compose.waitUntil(5_000) { earnings.earningsUiState.value is EarningsUiState.Success && earnings.transactionsState.value.items.size == 1 }
        compose.onNode(hasScrollAction()).performScrollToNode(hasText("View transactions (1)", substring = true))
        compose.onNodeWithText("View transactions (1)", substring = true).performSemanticsAction(SemanticsActions.OnClick)
        compose.onNodeWithContentDescription("Open purchase A1B2C3D4 by Aira S.").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitForIdle()
        compose.waitUntil(5_000) { purchases.previewState.value.selectedPurchase?.buyer == "Aira S." }
        val firstUrl = purchases.previewState.value.photo?.thumbnailUrl
        assertEquals("a1b2c3d4-order", purchases.previewState.value.orderId)
        assertEquals("tx-a", purchases.previewState.value.transactionId)

        restore.emulateSavedInstanceStateRestore()
        compose.waitForIdle()
        compose.waitUntil(5_000) { purchaseCalls.get() >= 2 && purchases.previewState.value.selectedPurchase?.buyer == "Bea T." }
        compose.onNodeWithText("Bea T.").assertExists()
        assertEquals("photo-a", purchases.previewState.value.photoId)
        assertEquals("a1b2c3d4-order", purchases.previewState.value.orderId)
        assertEquals("tx-a", purchases.previewState.value.transactionId)
        assertNotEquals(firstUrl, purchases.previewState.value.photo?.thumbnailUrl)
    }
}
