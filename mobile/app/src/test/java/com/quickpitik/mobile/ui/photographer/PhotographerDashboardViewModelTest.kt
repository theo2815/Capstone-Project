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
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList

/**
 * What the dashboard ViewModel still owns that a photographer can lose money
 * or work to: authoring an event, its coupon, and the fail-open approval
 * railguard every restricted action reads. Same construction as
 * PhotoManagementViewModelTest: the real ViewModel over a MockWebServer.
 *
 * Plan 2 phase G moved the rest out with the ViewModels that now own it -
 * payout accounts and the settings save to StudioSettingsViewModelTest, the
 * payout request to StudioEarningsViewModelTest.
 */
@RunWith(RobolectricTestRunner::class)
class PhotographerDashboardViewModelTest {
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
        // Every model() fires this from init. Without it two of the three tests
        // 404 on their own cold start and pass only because they never read it.
        route("GET $EVENTS") { ok("""{"items":[],"total":0,"offset":0,"limit":100}""") }
    }

    @After fun tearDown() {
        // No Room teardown: this ViewModel has held no AppDatabase reference
        // since the split (ce17a518), so there is nothing here to close.
        viewModels.clear()
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

    @Test fun `authoring an event reports the backend's words on a refusal and refreshes the list on a save`() {
        // The last two envelope sites phase E left in this file. A create and
        // an edit go to different endpoints; both fold through the same reader.
        route("POST $EVENTS") {
            MockResponse().setBody(error("EVENT_DATE_PAST", "Pick a date that hasn't passed."))
        }
        val vm = model()
        vm.eventsState.awaitFirst { it is EventsState.Success }

        vm.submitEvent(null, OwnedEventFormInput(title = "Cebu Run", date = "2020-01-01"))
        assertEquals(
            EventFormState.Error("Pick a date that hasn't passed."),
            vm.eventFormState.awaitFirst { it is EventFormState.Error },
        )

        route("POST $EVENTS") { ok(EVENT) }
        val listedBefore = count("GET", EVENTS)
        vm.submitEvent(null, OwnedEventFormInput(title = "Cebu Run", date = "2026-12-01"))
        vm.eventFormState.awaitFirst { it is EventFormState.Success }
        awaitUntil { count("GET", EVENTS) == listedBefore + 1 } // a saved event refreshes the roster

        // An edit PATCHes the event's own route; a thrown refusal still reads
        // as the backend's sentence, not "HTTP 409".
        route("PATCH $EVENTS/ev-1") {
            MockResponse().setResponseCode(409).setBody(error("EVENT_LOCKED", "This event is locked."))
        }
        vm.submitEvent("ev-1", OwnedEventFormInput(title = "Cebu Run 2"))
        assertEquals(
            EventFormState.Error("This event is locked."),
            vm.eventFormState.awaitFirst { it == EventFormState.Error("This event is locked.") },
        )
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
        const val COUPON = "/api/v1/me/photographer/events/ev-1/coupon"
        const val EVENTS = "/api/v1/me/photographer/events"
        const val EVENT = """{"id":"ev-1","slug":"cebu-run","name":"Cebu Run","date":"2026-12-01","location":"Cebu","state":"draft","visibility":"public","pricingMode":"flat","pricePerPhoto":50.0,"description":null,"organizerName":null,"watermarkPolicy":null,"photoCount":0,"bannerUrl":null}"""
        fun error(code: String, message: String) = """{"success":false,"errors":[{"code":"$code","message":"$message"}]}"""
        fun verification(status: String) = """{"status":"$status","missing":[],"missingCodes":[],"suspendedAt":null,"suspensionReason":null}"""
        fun coupon(code: String, percent: Int) =
            """{"eventId":"ev-1","code":"$code","percentOff":$percent,"active":true,"usageLimit":10,"usageCount":0,"updatedAt":"2026-09-11T00:00:00Z"}"""
    }
}
