package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.ViewModelStore
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.awaitRoomIdle
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList

/**
 * What the dashboard ViewModel still owns of the photographer's money: the
 * payout request's backend reasons, coupons, and the fail-open approval
 * railguard. Same construction as PhotoManagementViewModelTest: the real
 * ViewModel over a MockWebServer.
 *
 * The payout-account and settings-save cases moved to
 * StudioSettingsViewModelTest with plan 2 phase G.
 */
@RunWith(RobolectricTestRunner::class)
class PhotographerDashboardViewModelMoneyTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private val viewModels = ViewModelStore()
    private val seen = CopyOnWriteArrayList<Triple<String, String, String>>() // method, path, body
    private val routes = ConcurrentHashMap<String, (RecordedRequest) -> MockResponse>()

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val path = request.path.orEmpty().substringBefore('?')
                seen += Triple(request.method.orEmpty(), path, request.body.readUtf8())
                return routes["${request.method} $path"]?.invoke(request) ?: MockResponse().setResponseCode(404)
            }
        }
        route("GET /api/v1/me/photographer/verification") { ok(verification("approved")) }
    }

    @After fun tearDown() {
        viewModels.clear()
        awaitRoomIdle(application)
        AppDatabase.getDatabase(application).close()
        SessionManager.getInstance(application).clearSession()
    }

    private fun route(key: String, respond: (RecordedRequest) -> MockResponse) { routes[key] = respond }
    private fun model() = PhotographerDashboardViewModel(application).also { viewModels.put("photographer", it) }
    private fun count(method: String, path: String) = seen.count { it.first == method && it.second == path }
    private fun bodies(method: String, path: String) = seen.filter { it.first == method && it.second == path }.map { it.third }
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
        val earningsBefore = count("GET", "/api/v1/me/photographer/earnings")
        vm.submitPayoutRequest()
        vm.payoutActionState.awaitFirst { it == "Success: Payout request submitted successfully!" }
        awaitUntil { count("GET", "/api/v1/me/photographer/earnings") == earningsBefore + 1 }
    }

    @Test fun `coupons load, save, keep the loaded coupon through a refused save, and delete`() {
        route("GET $COUPON") { ok("null") }
        val vm = model()
        vm.loadCoupon("ev-1")
        assertEquals(CouponUiState.Loaded(null), vm.couponState.awaitFirst { it is CouponUiState.Loaded })

        route("PUT $COUPON") { ok(coupon("SAVE20", 20)) }
        vm.saveCoupon("ev-1", "SAVE20", 20, active = true, expiresAt = null, usageLimit = 10)
        val loaded = vm.couponState.awaitFirst { (it as? CouponUiState.Loaded)?.coupon != null } as CouponUiState.Loaded
        assertEquals("SAVE20", loaded.coupon?.code)
        assertTrue(bodies("PUT", COUPON).single().contains(""""usageLimit":10"""))

        route("PUT $COUPON") { MockResponse().setResponseCode(400).setBody(error("COUPON_PERCENT", "Percent must be 1-50.")) }
        vm.saveCoupon("ev-1", "SAVE99", 99, active = true, expiresAt = null, usageLimit = null)
        assertEquals("Percent must be 1-50.", vm.couponActionMessage.awaitFirst { it != null })
        assertEquals("SAVE20", (vm.couponState.value as CouponUiState.Loaded).coupon?.code)

        route("DELETE $COUPON") { ok("""{"removed":true}""") }
        vm.deleteCoupon("ev-1")
        vm.couponState.awaitFirst { (it as? CouponUiState.Loaded)?.coupon == null }
    }

    @Test fun `the approval railguard fails open on unknown status and blocks only a loaded non-approved one`() {
        route("GET /api/v1/me/photographer/verification") { MockResponse().setResponseCode(500) }
        val vm = model()
        vm.verificationState.awaitFirst { it is VerificationUiState.Error }
        assertTrue(vm.dashboardApproved.value)

        route("GET /api/v1/me/photographer/verification") { ok(verification("rejected")) }
        vm.fetchVerificationStatus()
        assertFalse(vm.dashboardApproved.awaitFirst { !it })

        route("GET /api/v1/me/photographer/verification") { ok(verification("approved")) }
        vm.fetchVerificationStatus()
        assertTrue(vm.dashboardApproved.value) // loading again: never blocks while the answer is unknown
        assertTrue(vm.dashboardApproved.awaitFirst { it })

        route("GET /api/v1/me/photographer/verification") { ok(verification("pending")) }
        vm.fetchVerificationStatus()
        assertFalse(vm.dashboardApproved.awaitFirst { !it })
    }

    private companion object {
        const val PAYOUTS = "/api/v1/me/photographer/payouts"
        const val COUPON = "/api/v1/me/photographer/events/ev-1/coupon"
        const val PAYOUT = """{"id":"py-1","weekOf":"2026-09-07","amount":500.0,"status":"pending","settledAt":null,"method":"gcash","reference":null,"holdReason":null}"""
        fun error(code: String, message: String) = """{"success":false,"errors":[{"code":"$code","message":"$message"}]}"""
        fun verification(status: String) = """{"status":"$status","missing":[],"missingCodes":[],"suspendedAt":null,"suspensionReason":null}"""
        fun coupon(code: String, percent: Int) =
            """{"eventId":"ev-1","code":"$code","percentOff":$percent,"active":true,"usageLimit":10,"usageCount":0,"updatedAt":"2026-09-11T00:00:00Z"}"""
    }
}
