package com.quickpitik.mobile

import android.app.Application
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
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
import androidx.lifecycle.viewmodel.compose.LocalViewModelStoreOwner
import androidx.lifecycle.ViewModelStore
import androidx.lifecycle.ViewModelStoreOwner
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
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.rules.TestRule
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.shadows.ShadowDialog
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicBoolean

@RunWith(RobolectricTestRunner::class)
class StudioShareRestorationTest {
    private val backend = MockBackend()
    private val compose = createComposeRule()
    @get:Rule val rules: TestRule = RuleChain.outerRule(backend).around(compose)
    private lateinit var application: Application
    private val purchaseCalls = AtomicInteger()
    private val eventCalls = AtomicInteger()
    private val requestedOrderId = AtomicReference<String?>()
    private val restoring = AtomicBoolean(false)
    private val failPinned = AtomicBoolean(false)
    private val manyPhotos = AtomicBoolean(false)

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                return when {
                    request.path.orEmpty().contains("/events/event-a?includeRemoved=true") -> {
                        eventCalls.incrementAndGet()
                        ok("""{"id":"event-a","slug":"cebu","name":"Cebu Run","date":"2026-10-01","location":"Cebu","state":"past","photoCount":1,"salesCount":1,"revenueKept":25.5,"deletedAt":null}""")
                    }
                    request.path.orEmpty().contains("purchased-photos") && manyPhotos.get() -> {
                        val items = (0 until 20).map { i ->
                            val id = if (i == 14) "photo-a" else "photo-$i"
                            """{"photo":{"id":"$id","bib":"$i","status":"ready","salesCount":1,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":null},"purchaseCount":1,"lastPurchasedAt":"2026-10-01T00:00:00Z"}"""
                        }
                        ok("""{"items":[${items.joinToString(",")}],"total":20,"offset":0,"limit":50}""")
                    }
                    request.path.orEmpty().contains("purchased-photos") -> ok("""{"items":[{"photo":{"id":"photo-a","bib":"123","status":"ready","salesCount":1,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":null},"purchaseCount":1,"lastPurchasedAt":"2026-10-01T00:00:00Z"}],"total":1,"offset":0,"limit":50}""")
                    request.path.orEmpty().contains("/photos/photo-a/purchases") -> {
                        val call = purchaseCalls.incrementAndGet()
                        val orderId = request.requestUrl?.queryParameter("orderId")
                        requestedOrderId.set(orderId)
                        if (failPinned.get()) return MockResponse().setResponseCode(404)
                        val signedUrl = backend.server.url("/signed-$call").toString()
                        val photo = """{"id":"photo-a","bib":"123","status":"ready","salesCount":1,"uploadedAt":"2026-10-01T00:00:00Z","tone":0,"span":"normal","thumbnailUrl":"$signedUrl","deletedAt":null,"deletionStartedAt":null}"""
                        fun receipt(id: String, buyer: String): String {
                            val refunded = id == "b1b2c3d4-order"
                            val transactions = if (refunded) """[{"id":"tx-sale","isRefund":false,"refundOf":null,"amountKept":25.5,"discountPhp":0,"paidAt":"2026-10-01T00:00:00Z","recordedAt":"2026-10-01T00:00:00Z"},{"id":"tx-a","isRefund":true,"refundOf":"tx-sale","amountKept":-25.5,"discountPhp":0,"paidAt":"2026-10-01T00:00:00Z","recordedAt":"2026-10-02T00:00:00Z"}]""" else "[]"
                            return """{"orderId":"$id","photoId":"photo-a","buyer":"$buyer","purchasedAt":"2026-10-01T00:00:00Z","orderCreatedAt":"2026-10-01T00:00:00Z","paymentMethod":"gcash","pricePhp":50.0,"discountPhp":0.0,"paidPhp":50.0,"kind":"paid","refundStatus":"${if (refunded) "full" else "none"}","refundPending":false,"refundedPhp":${if (refunded) 50 else 0},"refundedAt":${if (refunded) "\"2026-10-02T00:00:00Z\"" else "null"},"netKeptPhp":${if (refunded) 0 else 25.5},"transactions":$transactions}"""
                        }
                        val a = receipt("a1b2c3d4-order", "Aira S.")
                        val b = receipt("b1b2c3d4-order", if (restoring.get()) "Cora R." else "Bea T.")
                        val offset = request.requestUrl!!.queryParameter("offset")!!.toInt()
                        val items = if (offset == 0) listOf(a) + (2..50).map { receipt("order-$it", "Buyer $it") } else listOf(b)
                        ok("""{"photo":$photo,"event":{"id":"event-a","name":"Cebu Run","date":"2026-10-01","deletedAt":null},"items":[${items.joinToString(",")}],"total":51,"offset":$offset,"limit":50,"selectedPurchase":${if (orderId == "b1b2c3d4-order") b else a}}""")
                    }
                    request.path.orEmpty().contains("earnings/per-event") -> ok("""{"items":[],"total":0,"offset":0,"limit":10}""")
                    request.path.orEmpty().endsWith("/earnings") -> ok("""{"lifetimeKept":25.5,"thisWeek":25.5,"thisMonth":25.5,"payoutPending":0.0,"payoutScheduledFor":null,"weeklySeries":[],"thisWeekSold":1,"thisMonthSold":1}""")
                    request.path.orEmpty().contains("payouts/balance") -> ok("""{"unpaidBalance":0.0,"clearing":0.0,"reserved":0.0,"accountCooldownUntil":null,"minimum":100.0,"hasOpenRequest":false,"openRequest":null,"primaryAccount":null}""")
                    request.path.orEmpty().contains("billing/transactions") -> ok("""{"items":[{"id":"tx-a","paidAt":"2026-10-01T00:00:00Z","eventId":"event-a","eventName":"Cebu Run","eventSlug":"cebu","photoId":"photo-a","orderId":"b1b2c3d4-order","buyer":"Bea T.","amountKept":-25.5,"isRefund":true,"refundOf":"tx-sale","recordedAt":"2026-10-02T00:00:00Z","thumbnailUrl":null}],"total":1,"offset":0,"limit":50,"monthTotals":{"2026-10":-25.5}}""")
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }
    }

    @After fun tearDown() { SessionManager.getInstance(application).clearSession() }

    private fun newOwner() = object : ViewModelStoreOwner {
        override val viewModelStore = ViewModelStore()
    }

    @Test fun `NavHost restores photo id and refetches private detail then clears it on session change`() {
        lateinit var nav: NavHostController
        lateinit var purchases: PhotographerPurchasesViewModel
        var owner: ViewModelStoreOwner = newOwner()
        val restore = StateRestorationTester(compose)
        restore.setContent {
            DisposableEffect(Unit) {
                onDispose {
                    owner.viewModelStore.clear()
                    owner = newOwner()
                }
            }
            CompositionLocalProvider(LocalViewModelStoreOwner provides owner) {
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
        }
        compose.runOnIdle { nav.navigate(shareRoute("event-a", "purchased")) }
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.purchasedPhotosState.value.items.size == 1 }
        compose.waitForIdle()
        compose.onNodeWithText("1 purchase").performScrollTo().performSemanticsAction(SemanticsActions.OnClick)
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.selectedPurchase?.buyer == "Aira S." }
        compose.onNodeWithText("Aira S.").assertExists()
        compose.onNodeWithText("Load more purchases").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.items.size == 51 }
        compose.onNodeWithText("Bea T. · B1B2C3D4").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.selectedPurchase?.buyer == "Bea T." }
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); !purchases.previewState.value.isLoading }
        val firstVm = purchases
        val firstUrl = purchases.previewState.value.photo?.thumbnailUrl
        val callsBeforeRestore = purchaseCalls.get()
        assertTrue(callsBeforeRestore >= 1)

        restoring.set(true)
        val firstOwner = owner
        val firstGraph = nav.getBackStackEntry("studio")
        restore.emulateSavedInstanceStateRestore()
        compose.waitForIdle()
        assertNotSame(firstOwner, owner)
        assertNotSame(firstGraph, nav.getBackStackEntry("studio"))
        assertNotSame(firstVm, purchases)
        compose.waitUntil(5_000) { compose.waitForIdle(); purchaseCalls.get() > callsBeforeRestore && purchases.previewState.value.selectedPurchase?.buyer == "Cora R." }
        compose.waitForIdle()
        compose.onNodeWithText("Cora R.").assertExists()
        compose.onNodeWithText("Aira S.").assertDoesNotExist()
        assertNotEquals(firstUrl, purchases.previewState.value.photo?.thumbnailUrl)
        assertTrue(eventCalls.get() >= 2)
        assertEquals("photo-a", purchases.previewState.value.photoId)
        assertEquals("b1b2c3d4-order", purchases.previewState.value.orderId)
        assertEquals("b1b2c3d4-order", requestedOrderId.get())
        assertEquals(50, purchases.previewState.value.items.size)
        assertFalse(purchases.previewState.value.items.any { it.orderId == "b1b2c3d4-order" })
        compose.onNodeWithText("B1B2C3D4").assertExists()

        failPinned.set(true)
        compose.runOnIdle { purchases.retryPreview() }
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.errorStatus == 404 }
        compose.waitForIdle()
        compose.onNodeWithText("Purchase unavailable.").assertExists()
        compose.onNodeWithText("Aira S.").assertDoesNotExist()
        assertEquals(null, purchases.previewState.value.selectedPurchase)
        failPinned.set(false)
        val callsBeforeRetry = purchaseCalls.get()
        compose.onNodeWithText("Retry purchases").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.selectedPurchase?.buyer == "Cora R." }
        assertEquals(callsBeforeRetry + 1, purchaseCalls.get())
        assertEquals("b1b2c3d4-order", requestedOrderId.get())

        SessionManager.getInstance(application).saveSession("other", "RUNNER", "Other", "other@x.com")
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.photoId == null }
        compose.waitForIdle()
        compose.onNodeWithText("Cora R.").assertDoesNotExist()
    }

    @Test fun `cold restore in a multi-photo gallery keeps the exact photo and order`() {
        manyPhotos.set(true)
        lateinit var nav: NavHostController
        lateinit var purchases: PhotographerPurchasesViewModel
        var owner: ViewModelStoreOwner = newOwner()
        val restore = StateRestorationTester(compose)
        restore.setContent {
            DisposableEffect(Unit) {
                onDispose {
                    owner.viewModelStore.clear()
                    owner = newOwner()
                }
            }
            CompositionLocalProvider(LocalViewModelStoreOwner provides owner) {
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
                            purchases = viewModel(graph)
                            StudioShareDestination(
                                eventId = entry.arguments?.getString("eventId"),
                                initialTab = entry.arguments?.getString("tab") ?: "active",
                                dashboard = viewModel<PhotographerDashboardViewModel>(graph),
                                purchases = purchases,
                                onBack = { nav.popBackStack() },
                                onOpenEvent = {},
                            )
                        }
                    }
                }
            }
        }
        compose.runOnIdle { nav.navigate(shareRoute("event-a", "purchased")) }
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.purchasedPhotosState.value.items.size == 20 }
        compose.runOnIdle { purchases.openPurchase("photo-a") }
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.selectedPurchase?.buyer == "Aira S." }
        compose.onNodeWithText("15 / 20").assertExists()
        compose.onNodeWithText("Load more purchases").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.items.size == 51 }
        compose.onNodeWithText("Bea T. · B1B2C3D4").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.selectedPurchase?.buyer == "Bea T." && !purchases.previewState.value.isLoading }

        restoring.set(true)
        val firstVm = purchases
        restore.emulateSavedInstanceStateRestore()
        compose.waitForIdle()
        assertNotSame(firstVm, purchases)
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.purchasedPhotosState.value.items.size == 20 && purchases.previewState.value.selectedPurchase != null }
        compose.waitForIdle()
        assertEquals("photo-a", purchases.previewState.value.photoId)
        assertEquals("b1b2c3d4-order", purchases.previewState.value.orderId)
        assertEquals("b1b2c3d4-order", requestedOrderId.get())
        compose.onNodeWithText("Cora R.").assertExists()
        compose.onNodeWithText("15 / 20").assertExists()
    }

    @Test fun `NavHost restores transaction ids and refetches their buyer and URL`() {
        lateinit var nav: NavHostController
        lateinit var earnings: StudioEarningsViewModel
        lateinit var purchases: PhotographerPurchasesViewModel
        var owner: ViewModelStoreOwner = newOwner()
        val restore = StateRestorationTester(compose)
        restore.setContent {
            DisposableEffect(Unit) {
                onDispose {
                    owner.viewModelStore.clear()
                    owner = newOwner()
                }
            }
            CompositionLocalProvider(LocalViewModelStoreOwner provides owner) {
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
        }
        compose.runOnIdle { nav.navigate("studio/earnings") }
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); earnings.earningsUiState.value is EarningsUiState.Success && earnings.transactionsState.value.items.size == 1 }
        compose.onNode(hasScrollAction()).performScrollToNode(hasText("View transactions (1)", substring = true))
        compose.onNodeWithText("View transactions (1)", substring = true).performSemanticsAction(SemanticsActions.OnClick)
        compose.onNodeWithContentDescription("Open purchase B1B2C3D4 by Bea T.").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.selectedPurchase?.buyer == "Bea T." }
        compose.onNodeWithText("Load more purchases").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.items.size == 51 }
        compose.onNodeWithText("Aira S. · A1B2C3D4").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitForIdle()
        compose.onNodeWithText("Bea T. · B1B2C3D4").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.selectedPurchase?.buyer == "Bea T." }
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); !purchases.previewState.value.isLoading }
        val firstUrl = purchases.previewState.value.photo?.thumbnailUrl
        val callsBeforeRestore = purchaseCalls.get()
        assertEquals("b1b2c3d4-order", purchases.previewState.value.orderId)
        assertEquals("tx-a", purchases.previewState.value.transactionId)

        restoring.set(true)
        val firstVm = purchases
        val firstOwner = owner
        val firstGraph = nav.getBackStackEntry("studio")
        restore.emulateSavedInstanceStateRestore()
        compose.waitForIdle()
        assertNotSame(firstOwner, owner)
        assertNotSame(firstGraph, nav.getBackStackEntry("studio"))
        assertNotSame(firstVm, purchases)
        compose.waitUntil(5_000) { compose.waitForIdle(); purchaseCalls.get() > callsBeforeRestore && purchases.previewState.value.selectedPurchase?.buyer == "Cora R." }
        compose.onNodeWithText("Cora R.").assertExists()
        assertEquals("photo-a", purchases.previewState.value.photoId)
        assertEquals("b1b2c3d4-order", purchases.previewState.value.orderId)
        assertEquals("tx-a", purchases.previewState.value.transactionId)
        assertEquals("b1b2c3d4-order", requestedOrderId.get())
        assertNotEquals(firstUrl, purchases.previewState.value.photo?.thumbnailUrl)
        assertEquals(50, purchases.previewState.value.items.size)
        assertFalse(purchases.previewState.value.items.any { it.orderId == "b1b2c3d4-order" })
        compose.onNodeWithText("B1B2C3D4").assertExists()
        compose.onNodeWithText("Selected · Earnings adjustment", substring = true).assertExists()

        compose.onNodeWithText("Show Metadata").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitForIdle()
        compose.onNodeWithText("Photo metadata").assertExists()
        compose.runOnIdle { ShadowDialog.getShownDialogs().last { it.isShowing }.onBackPressed() }
        compose.waitForIdle()
        compose.onNodeWithText("Photo metadata").assertDoesNotExist()
        compose.onNodeWithContentDescription("Close preview").assertExists()
        compose.runOnIdle { ShadowDialog.getShownDialogs().last { it.isShowing }.onBackPressed() }
        compose.waitForIdle()
        compose.onNodeWithText("Recent sales history", ignoreCase = true).assertExists()
        compose.runOnIdle { ShadowDialog.getShownDialogs().last { it.isShowing }.onBackPressed() }
        compose.waitForIdle()
        compose.onNodeWithText("Recent sales history", ignoreCase = true).assertDoesNotExist()

        compose.onNode(hasScrollAction()).performScrollToNode(hasText("View transactions (1)", substring = true))
        compose.onNodeWithText("View transactions (1)", substring = true).performSemanticsAction(SemanticsActions.OnClick)
        compose.onNodeWithContentDescription("Open purchase B1B2C3D4 by Bea T.").performSemanticsAction(SemanticsActions.OnClick)
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.selectedPurchase != null }
        SessionManager.getInstance(application).saveSession("other", "PHOTOGRAPHER", "Other", "other@x.com")
        compose.waitUntil(5_000) { compose.waitForIdle(); purchases.previewState.value.photoId == null }
        compose.waitForIdle()
        compose.onNodeWithText("Cora R.").assertDoesNotExist()
        val callsAfterAccountChange = purchaseCalls.get()
        restore.emulateSavedInstanceStateRestore()
        compose.waitForIdle()
        compose.waitUntil(5_000) { compose.waitForIdle(); earnings.earningsUiState.value is EarningsUiState.Success }
        compose.waitForIdle()
        assertEquals(null, purchases.previewState.value.photoId)
        assertEquals(callsAfterAccountChange, purchaseCalls.get())
    }
}
