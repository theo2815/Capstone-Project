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
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * The payout cycle: a refused request must show the backend's own reason
 * ("you need at least 500 pesos"), not "HTTP 409 Conflict", and a granted one
 * must refresh the wallet so the balance the photographer is looking at is the
 * one the request just changed.
 *
 * Moved from PhotographerDashboardViewModelTest with the ViewModel split
 * (plan 2 phase G).
 */
@RunWith(RobolectricTestRunner::class)
class StudioEarningsViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private val viewModels = ViewModelStore()
    private val seen = CopyOnWriteArrayList<Pair<String, String>>() // method, path
    private val routes = ConcurrentHashMap<String, (RecordedRequest) -> MockResponse>()

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val path = request.path.orEmpty().substringBefore('?')
                seen += request.method.orEmpty() to path
                return routes["${request.method} $path"]?.invoke(request) ?: MockResponse().setResponseCode(404)
            }
        }
        // The whole cold start: three routes, all of this ViewModel's own.
        route("GET $EARNINGS") { ok(OVERVIEW) }
        route("GET $PAYOUTS/balance") { ok(BALANCE) }
        route("GET $LEDGER_PATH") { ok(LEDGER) }
    }

    @After fun tearDown() {
        viewModels.clear()
        SessionManager.getInstance(application).clearSession()
    }

    private fun route(key: String, respond: (RecordedRequest) -> MockResponse) { routes[key] = respond }
    private fun model() = StudioEarningsViewModel(application).also { viewModels.put("earnings", it) }
    private fun count(method: String, path: String) = seen.count { it.first == method && it.second == path }
    private fun awaitUntil(check: () -> Boolean) {
        val deadline = System.currentTimeMillis() + 5_000
        while (!check() && System.currentTimeMillis() < deadline) Thread.sleep(5)
        assertTrue(check())
    }

    @Test fun `a payout request surfaces the backend's reason, and a success refreshes the wallet`() {
        route("POST $PAYOUTS/request") {
            MockResponse().setResponseCode(409).setBody(error("PAYOUT_BELOW_MINIMUM", "You need at least 500 pesos to request a payout."))
        }
        val vm = model()
        vm.submitPayoutRequest()
        assertEquals(
            "Error: You need at least 500 pesos to request a payout.",
            vm.payoutActionState.awaitFirst { it != null && it != "processing" },
        )

        route("POST $PAYOUTS/request") { ok(PAYOUT) }
        val earningsBefore = count("GET", EARNINGS)
        vm.submitPayoutRequest()
        vm.payoutActionState.awaitFirst { it == "Success: Payout request submitted successfully!" }
        awaitUntil { count("GET", EARNINGS) == earningsBefore + 1 }
    }

    @Test fun `a withdrawn request says so and refreshes the wallet, and a refusal keeps the backend's words`() {
        route("POST $PAYOUTS/cy-1/withdraw") {
            MockResponse().setResponseCode(409).setBody(error("PAYOUT_NOT_HELD", "That request is already being paid out."))
        }
        val vm = model()
        vm.earningsUiState.awaitFirst { it is EarningsUiState.Success }

        vm.withdrawPayoutRequest("cy-1")
        assertEquals(
            "Error: That request is already being paid out.",
            vm.payoutActionState.awaitFirst { it != null && it != "processing" },
        )

        route("POST $PAYOUTS/cy-1/withdraw") { ok(PAYOUT) }
        val earningsBefore = count("GET", EARNINGS)
        vm.withdrawPayoutRequest("cy-1")
        vm.payoutActionState.awaitFirst {
            it == "Success: Request withdrawn — you can file a new one when ready."
        }
        awaitUntil { count("GET", EARNINGS) == earningsBefore + 1 }

        vm.clearPayoutActionState()
        assertEquals(null, vm.payoutActionState.value)
    }

    @Test fun `overview succeeds despite ledger failure`() {
        route("GET $LEDGER_PATH") {
            MockResponse().setBody(error("LEDGER_DOWN", "Ledger is unavailable."))
        }
        val vm = model()
        vm.earningsUiState.awaitFirst { it is EarningsUiState.Success }
        vm.transactionsState.awaitFirst { it.error != null }
        assertEquals("Ledger is unavailable.", vm.transactionsState.value.error)
    }

    @Test fun `logout discards an in-flight payout request result`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        route("POST $PAYOUTS/request") {
            started.countDown()
            release.await(5, TimeUnit.SECONDS)
            ok(PAYOUT)
        }
        val vm = model()
        vm.submitPayoutRequest()
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try {
            SessionManager.getInstance(application).clearSession()
        } finally { release.countDown() }
        Thread.sleep(250)
        assertNull(vm.payoutActionState.value)
        assertTrue(vm.earningsUiState.value is EarningsUiState.Loading)
    }

    @Test fun `role change discards an in-flight payout withdrawal result`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        route("POST $PAYOUTS/cy-1/withdraw") {
            started.countDown()
            release.await(5, TimeUnit.SECONDS)
            ok(PAYOUT)
        }
        val vm = model()
        vm.withdrawPayoutRequest("cy-1")
        assertTrue(started.await(5, TimeUnit.SECONDS))
        try {
            SessionManager.getInstance(application).saveSession("new-token", "RUNNER", "New", "new@x.com")
        } finally { release.countDown() }
        Thread.sleep(250)
        assertNull(vm.payoutActionState.value)
        assertTrue(vm.earningsUiState.value is EarningsUiState.Loading)
    }

    @Test fun `loads transaction 51 and preserves rows on retry`() {
        var failAppend = true
        route("GET $LEDGER_PATH") { request ->
            val offset = request.requestUrl?.queryParameter("offset")?.toInt() ?: 0
            if (offset == 50 && failAppend) MockResponse().setResponseCode(503)
            else ok(ledgerPage(offset, if (offset == 0) 50 else 25, 75))
        }
        val vm = model()
        vm.transactionsState.awaitFirst { it.items.size == 50 }
        vm.loadMoreTransactions()
        vm.transactionsState.awaitFirst { it.appendError != null }
        assertEquals(50, vm.transactionsState.value.items.size)
        assertEquals(50, vm.transactionsState.value.nextOffset)
        failAppend = false
        vm.loadMoreTransactions()
        vm.transactionsState.awaitFirst { it.items.size == 75 }
        assertEquals(75L, vm.transactionsState.value.total)
        assertFalse(vm.transactionsState.value.hasMore)
    }

    @Test fun `loads giveaway-only event earnings`() {
        route("GET $EARNINGS/per-event") { ok("""{"items":[{"eventId":"giveaway","eventName":"Fun Run","eventDate":"2026-10-01","deletedAt":null,"photoCount":1,"salesCount":0,"purchaseCount":3,"revenueKept":0}],"total":1,"offset":0,"limit":10}""") }
        val vm = model()
        vm.eventEarningsState.awaitFirst { it.items.size == 1 }
        assertEquals(3L, vm.eventEarningsState.value.items.single().purchaseCount)
        assertEquals(0.0, vm.eventEarningsState.value.items.single().revenueKept, 0.0)
    }

    @Test fun `initial ledger failure needs retry instead of an offset-zero append`() {
        var calls = 0
        route("GET $LEDGER_PATH") {
            calls++
            if (calls == 1) MockResponse().setResponseCode(503)
            else ok(ledgerPage(0, 1, 1))
        }
        val vm = model()
        vm.transactionsState.awaitFirst { it.error != null }
        assertFalse(vm.transactionsState.value.hasMore)
        vm.loadMoreTransactions()
        assertEquals(1, calls)
        vm.retryTransactions()
        vm.transactionsState.awaitFirst { it.items.size == 1 }
        assertEquals("tx-0", vm.transactionsState.value.items.single().id)
        assertNull(vm.transactionsState.value.error)
    }

    @Test fun `initial event earnings failure needs retry instead of an offset-zero append`() {
        var calls = 0
        route("GET $EARNINGS/per-event") {
            calls++
            if (calls == 1) MockResponse().setResponseCode(503)
            else ok("""{"items":[{"eventId":"event-a","eventName":"Run","eventDate":"2026-10-01","deletedAt":null,"photoCount":1,"salesCount":0,"purchaseCount":1,"revenueKept":0}],"total":1,"offset":0,"limit":10}""")
        }
        val vm = model()
        vm.eventEarningsState.awaitFirst { it.error != null }
        assertFalse(vm.eventEarningsState.value.hasMore)
        vm.loadMoreEventEarnings()
        assertEquals(1, calls)
        vm.retryEventEarnings()
        vm.eventEarningsState.awaitFirst { it.items.size == 1 }
        assertEquals("event-a", vm.eventEarningsState.value.items.single().eventId)
        assertNull(vm.eventEarningsState.value.error)
    }

    private fun ledgerPage(offset: Int, count: Int, total: Int): String =
        """{"items":[${(offset until offset + count).joinToString(",") { n -> """{"id":"tx-$n","paidAt":"2026-10-01T00:00:00Z","eventId":"event","eventName":"Run","eventSlug":"run","photoId":"photo-$n","orderId":"order-$n","buyer":"Aira S.","amountKept":1,"isRefund":false,"refundOf":null,"recordedAt":"2026-10-01T00:00:00Z","thumbnailUrl":null}""" }}],"total":$total,"offset":$offset,"limit":50,"monthTotals":{}}"""
    private companion object {
        const val PAYOUTS = "/api/v1/me/photographer/payouts"
        const val EARNINGS = "/api/v1/me/photographer/earnings"
        const val LEDGER_PATH = "/api/v1/me/photographer/billing/transactions"
        const val OVERVIEW = """{"lifetimeKept":1000.0,"thisWeek":100.0,"thisMonth":400.0,"payoutPending":500.0,"payoutScheduledFor":null,"weeklySeries":[],"thisWeekSold":1,"thisMonthSold":4}"""
        const val BALANCE = """{"unpaidBalance":500.0,"minimum":500.0,"hasOpenRequest":false,"openRequest":null}"""
        const val LEDGER = """{"items":[],"total":0,"offset":0,"limit":50,"monthTotals":{}}"""
        const val PAYOUT = """{"id":"py-1","weekOf":"2026-09-07","amount":500.0,"status":"pending","settledAt":null,"method":"gcash","reference":null,"holdReason":null}"""
        fun error(code: String, message: String) = """{"success":false,"errors":[{"code":"$code","message":"$message"}]}"""
    }
}
