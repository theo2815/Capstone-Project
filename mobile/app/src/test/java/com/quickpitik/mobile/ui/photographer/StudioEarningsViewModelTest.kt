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
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList

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

    @Test fun `one refused endpoint fails the whole tab with that endpoint's reason`() {
        // The three loads are fired together; the ledger's refusal must reach
        // the screen rather than an empty Success built from two thirds of it.
        route("GET $LEDGER_PATH") {
            MockResponse().setBody(error("LEDGER_DOWN", "Ledger is unavailable."))
        }
        val state = model().earningsUiState.awaitFirst { it is EarningsUiState.Error }
        assertEquals("Ledger is unavailable.", (state as EarningsUiState.Error).message)
    }

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
