package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.ViewModelStore
import androidx.test.core.app.ApplicationProvider
import androidx.work.Configuration
import androidx.work.WorkManager
import androidx.work.testing.SynchronousExecutor
import androidx.work.testing.WorkManagerTestInitHelper
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import com.quickpitik.mobile.worker.PhotoUploadScheduler
import kotlinx.coroutines.runBlocking
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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.TimeUnit

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
        WorkManagerTestInitHelper.initializeTestWorkManager(application,
            Configuration.Builder().setExecutor(SynchronousExecutor()).build())
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
        WorkManagerTestInitHelper.closeWorkDatabase()
        SessionManager.getInstance(application).clearSession()
    }

    private fun route(key: String, respond: (RecordedRequest) -> MockResponse) { routes[key] = respond }
    private fun model() = PhotographerDashboardViewModel(application).also { viewModels.put("photographer", it) }
    private fun count(method: String, path: String) = seen.count { it.first == method && it.second == path }
    private fun bodies(method: String, path: String) = seen.filter { it.first == method && it.second == path }.map { it.third }
    private fun uploadWork() =
        WorkManager.getInstance(application).getWorkInfosForUniqueWork(PhotoUploadScheduler.WORK_NAME).get()
    private fun awaitUntil(check: () -> Boolean) {
        val deadline = System.currentTimeMillis() + 5_000
        while (!check() && System.currentTimeMillis() < deadline) Thread.sleep(5)
        assertTrue(check())
    }

    @Test fun `race types come from the server and an absent switch stays off`() {
        route("GET /api/v1/race-types") { ok("""{"selfServeEnabled":true,"types":[{"code":"ROAD_RACE","label":"Road race","hint":"On roads","shared":true},{"code":"OTHER","label":"Other","hint":"Your own event","shared":false}]}""") }
        val vm = model()
        vm.loadRaceTypes()
        val types = vm.raceTypes.awaitFirst { it?.selfServeEnabled == true }!!
        assertEquals("ROAD_RACE", types.types?.first()?.code)
        route("GET /api/v1/race-types") { ok("{}") }
        vm.loadRaceTypes()
        vm.raceTypes.awaitFirst { it?.selfServeEnabled != true }
    }

    @Test fun `duplicate answer moves to Duplicate state`() {
        val vm = model()
        for (outcome in listOf("match", "candidates")) {
            route("POST $RACES") { ok(duplicate(outcome)) }
            vm.submitRace(raceInput())
            val state = vm.raceFormState.awaitFirst { it is RaceFormState.Duplicate } as RaceFormState.Duplicate
            assertEquals(outcome, state.outcome)
            assertEquals("ev-1", state.candidates.single().id)
            assertEquals("pending", state.candidates.single().reviewStatus)
            vm.dismissRaceDuplicate()
        }
    }

    @Test fun `mine is different on a match moves to PendingReview`() {
        route("POST $RACES") { ok(duplicate("match")) }
        route("GET /api/v1/events") { ok("""{"items":[],"total":0,"offset":0,"limit":100}""") }
        val vm = model()
        vm.submitRace(raceInput())
        vm.raceFormState.awaitFirst { it is RaceFormState.Duplicate }
        route("POST $RACES") { ok("""{"outcome":"pending_review","event":$PENDING_RACE,"candidates":[],"checkReasons":["POSSIBLE_DUPLICATE"]}""") }
        vm.submitRace(raceInput(), confirmDifferentFrom = "ev-1")
        val result = vm.raceFormState.awaitFirst { it is RaceFormState.PendingReview } as RaceFormState.PendingReview
        assertEquals("pending", result.result.event?.reviewStatus)
        val sent = bodies("POST", RACES).last()
        for (field in listOf("regionCode", "provinceCode", "eventType", "confirmDifferentFrom")) {
            assertTrue(field, sent.contains("name=\"$field\""))
        }
        assertTrue(sent.contains("ev-1"))
        for (field in listOf("pricePerPhoto", "pricingMode", "watermarkPolicy", "visibility")) {
            assertFalse(field, sent.contains("name=\"$field\""))
        }
    }

    @Test fun `joining a pending race explains the wait and refreshes both rosters`() {
        route("POST $RACES/ev-1/join") { ok("""{"outcome":"joined","event":$PENDING_RACE,"candidates":[],"checkReasons":[]}""") }
        route("GET /api/v1/events") { ok("""{"items":[],"total":0,"offset":0,"limit":100}""") }
        val vm = model()
        vm.eventsState.awaitFirst { it is EventsState.Success }
        val before = count("GET", EVENTS)
        vm.joinRace("ev-1", "Cebu Run")
        vm.raceFormState.awaitFirst { it is RaceFormState.PendingReview }
        assertEquals("""{"typedName":"Cebu Run"}""", bodies("POST", "$RACES/ev-1/join").single())
        awaitUntil { count("GET", EVENTS) == before + 1 && count("GET", "/api/v1/events") == 1 }
    }

    @Test fun `a refused duplicate choice keeps its candidates for retry`() {
        route("POST $RACES") { ok(duplicate("candidates")) }
        val vm = model()
        vm.submitRace(raceInput())
        vm.raceFormState.awaitFirst { it is RaceFormState.Duplicate }
        route("POST $RACES/ev-1/join") { MockResponse().setResponseCode(409).setBody(error("RACE_HIDDEN", "This race is hidden.")) }
        vm.joinRace("ev-1", "Cebu Run")
        val state = vm.raceFormState.awaitFirst { (it as? RaceFormState.Duplicate)?.error != null } as RaceFormState.Duplicate
        assertEquals("This race is hidden.", state.error)
        assertEquals("ev-1", state.candidates.single().id)
    }

    @Test fun `a newly created race moves to Created and a malformed result stays on the form`() {
        route("POST $RACES") { ok("""{"outcome":"created","event":$EVENT,"candidates":[],"checkReasons":[]}""") }
        route("GET /api/v1/events") { ok("""{"items":[],"total":0,"offset":0,"limit":100}""") }
        val vm = model()
        vm.submitRace(raceInput())
        vm.raceFormState.awaitFirst { it is RaceFormState.Created }
        vm.resetEventForm()
        route("POST $RACES") { ok("{}") }
        vm.submitRace(raceInput())
        vm.raceFormState.awaitFirst { it is RaceFormState.Error }
    }

    private fun raceInput() = OwnedEventFormInput(title = "Cebu Run", date = "2026-11-11", location = "Cebu",
        eventType = "ROAD_RACE", regionCode = "07", provinceCode = "0722", pricingMode = "paid", pricePerPhoto = "500", visibility = "unlisted")

    @Test fun `resetting a form prevents an old race reply from replacing its new state`() {
        route("POST $RACES") { ok(duplicate("match")).setBodyDelay(400, TimeUnit.MILLISECONDS) }
        val vm = model()
        vm.submitRace(raceInput())
        awaitUntil { count("POST", RACES) == 1 }
        vm.resetEventForm()
        Thread.sleep(600)
        assertEquals(RaceFormState.Idle, vm.raceFormState.value)
    }

    // The upload roster must keep the admin-only list: another photographer's
    // admin-listed event is not uploadable here.
    @Test fun `the public events roster never asks for admin-listed events`() {
        val requested = CopyOnWriteArrayList<String>()
        route("GET /api/v1/events") { requested += it.path.orEmpty(); ok("""{"items":[],"total":0,"offset":0,"limit":100}""") }
        model().fetchPublicEvents()
        awaitUntil { requested.isNotEmpty() }
        assertFalse(requested.single().contains("includeListed"))
    }

    @Test fun `event metadata visibility loads selected event, cancels, and saves its own setting`() {
        val path = "$EVENTS/ev-1/metadata-visibility"
        route("GET $path") { ok("""{"metadataPublic":false}""") }
        val vm = model()
        vm.loadEventMetadataVisibility("ev-1")
        vm.eventMetadataVisibility.awaitFirst { it.eventId == "ev-1" && it.saved == false }
        vm.setEventMetadataVisibilityDraft(true)
        vm.cancelEventMetadataVisibility()
        assertEquals(false, vm.eventMetadataVisibility.value.draft)
        vm.setEventMetadataVisibilityDraft(true)
        route("PUT $path") { ok("""{"metadataPublic":true}""") }
        vm.saveEventMetadataVisibility()
        vm.eventMetadataVisibility.awaitFirst { it.saved == true }
        assertEquals("""{"metadataPublic":true}""", bodies("PUT", path).single())
    }

    @Test fun `event metadata visibility failure keeps draft and an event switch loads its own preference`() {
        val first = "$EVENTS/ev-1/metadata-visibility"
        val second = "$EVENTS/ev-2/metadata-visibility"
        route("GET $first") { ok("""{"metadataPublic":false}""") }
        route("GET $second") { ok("""{"metadataPublic":true}""") }
        val vm = model()
        vm.loadEventMetadataVisibility("ev-1")
        vm.eventMetadataVisibility.awaitFirst { it.eventId == "ev-1" && it.saved == false }
        vm.setEventMetadataVisibilityDraft(true)
        route("PUT $first") { MockResponse().setResponseCode(500) }
        vm.saveEventMetadataVisibility()
        vm.eventMetadataVisibility.awaitFirst { it.error != null }
        assertEquals(true, vm.eventMetadataVisibility.value.draft)
        vm.loadEventMetadataVisibility("ev-2")
        vm.eventMetadataVisibility.awaitFirst { it.eventId == "ev-2" && it.saved == true }
        assertEquals(true, vm.eventMetadataVisibility.value.draft)
        assertEquals(1, count("GET", second))
        assertEquals(0, count("PUT", second))
    }

    @Test fun `late old event visibility does not replace selected event`() {
        val first = "$EVENTS/ev-1/metadata-visibility"
        val second = "$EVENTS/ev-2/metadata-visibility"
        route("GET $first") { ok("""{"metadataPublic":true}""").setBodyDelay(300, TimeUnit.MILLISECONDS) }
        route("GET $second") { ok("""{"metadataPublic":false}""") }
        val vm = model()
        vm.loadEventMetadataVisibility("ev-1")
        awaitUntil { count("GET", first) == 1 }
        vm.loadEventMetadataVisibility("ev-2")
        vm.eventMetadataVisibility.awaitFirst { it.eventId == "ev-2" && it.saved == false }
        Thread.sleep(400)
        assertEquals("ev-2", vm.eventMetadataVisibility.value.eventId)
        assertEquals(false, vm.eventMetadataVisibility.value.saved)
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

        val approvedReply = CountDownLatch(1)
        route("GET /api/v1/me/photographer/verification") {
            approvedReply.await()
            ok(verification("approved"))
        }
        try {
            vm.fetchVerificationStatus()
            vm.verificationState.awaitFirst { it is VerificationUiState.Loading }
            assertTrue(vm.dashboardApproved.awaitFirst { it }) // loading never blocks while the answer is unknown
        } finally {
            approvedReply.countDown()
        }
        vm.verificationState.awaitFirst { it is VerificationUiState.Success && it.verification.status == "approved" }
        assertTrue(vm.dashboardApproved.awaitFirst { it })

        route("GET /api/v1/me/photographer/verification") { ok(verification("pending")) }
        vm.fetchVerificationStatus()
        assertFalse(vm.dashboardApproved.awaitFirst { !it })
    }

    @Test fun `a legacy seller-details response leaves the additive fields null`() = runBlocking {
        route("GET /api/v1/me/photographer/seller-details") { ok("""{"status":"submitted","rejectionReason":null}""") }
        val details = requireNotNull(RetrofitClient.apiService.getSellerDetails("Bearer t").data)
        assertEquals("submitted", details.status)
        assertNull(details.reviewState)
        assertNull(details.uploadsAllowed)
        assertNull(details.identityGateStartsOn)
    }

    @Test fun `a manual-review response takes precedence over its legacy submitted status`() {
        route("GET /api/v1/me/photographer/seller-details") {
            ok("""{"status":"submitted","reviewState":"manual_review","uploadsAllowed":false,"identityGateStartsOn":"2026-11-17","rejectionReason":null}""")
        }
        assertEquals(
            SellerDetailsCard.ManualReview(paused = true, dueDate = null),
            model().sellerDetailsState.awaitFirst { it != null },
        )
    }

    @Test fun `an answer that allows uploads enqueues the upload work`() {
        route("GET /api/v1/me/photographer/seller-details") {
            ok("""{"status":"approved","reviewState":"approved","uploadsAllowed":true,"identityGateStartsOn":null,"rejectionReason":null}""")
        }
        model()
        awaitUntil { uploadWork().isNotEmpty() }
    }

    @Test fun `a paused answer enqueues nothing`() {
        route("GET /api/v1/me/photographer/seller-details") {
            ok("""{"status":"submitted","reviewState":"manual_review","uploadsAllowed":false,"identityGateStartsOn":null,"rejectionReason":null}""")
        }
        model().sellerDetailsState.awaitFirst { it != null }
        Thread.sleep(300) // the enqueue would land on the scheduler's IO thread just after the state is set
        assertTrue(uploadWork().isEmpty())
    }

    @Test fun `a legacy seller state still shows its rejection card`() {
        route("GET /api/v1/me/photographer/seller-details") { ok("""{"status":"rejected","rejectionReason":"Expired ID."}""") }
        assertEquals(
            SellerDetailsCard.Rejected("Expired ID"),
            model().sellerDetailsState.awaitFirst { it != null },
        )
    }

    @Test fun `an approved profile submit says the studio is live`() {
        route("POST /api/v1/me/photographer/verification") { ok(verification("approved")) }
        val vm = model()
        vm.submitVerification()
        assertEquals(
            ActionState.Done("Your studio is live."),
            vm.verificationActionState.awaitFirst { it is ActionState.Done || it is ActionState.Failed },
        )
    }

    @Test fun `a pending profile submit explains the identity check`() {
        route("POST /api/v1/me/photographer/verification") { ok(verification("pending")) }
        val vm = model()
        vm.submitVerification()
        assertEquals(
            ActionState.Done("Saved. Your studio goes live when your identity check passes."),
            vm.verificationActionState.awaitFirst { it is ActionState.Done || it is ActionState.Failed },
        )
    }

    @Test fun `a held profile submit says an admin is reviewing it`() {
        route("POST /api/v1/me/photographer/verification") {
            ok(verification("pending").replace("}", ""","awaitingAdmin":true}"""))
        }
        val vm = model()
        vm.submitVerification()
        assertEquals(
            ActionState.Done("Saved. An admin is reviewing your profile."),
            vm.verificationActionState.awaitFirst { it is ActionState.Done || it is ActionState.Failed },
        )
    }

    private companion object {
        const val RACES = "/api/v1/me/photographer/races"
        const val PENDING_RACE = """{"id":"ev-1","slug":"cebu-run","name":"Cebu Run","date":"2026-11-11","location":"Cebu","state":"draft","reviewStatus":"pending","ownedByMe":false,"photoCount":0,"salesCount":0,"revenueKept":0}"""
        fun duplicate(outcome: String) = """{"outcome":"$outcome","event":null,"candidates":[{"id":"ev-1","slug":"cebu-run","name":"Cebu Run","date":"2026-11-11","location":"Cebu","regionCode":"07","provinceCode":"0722","organizerName":"Club","eventType":"ROAD_RACE","bannerUrl":null,"reviewStatus":"pending","strength":"strong"}],"checkReasons":[]}"""
        const val COUPON = "/api/v1/me/photographer/events/ev-1/coupon"
        const val EVENTS = "/api/v1/me/photographer/events"
        const val EVENT = """{"id":"ev-1","slug":"cebu-run","name":"Cebu Run","date":"2026-12-01","location":"Cebu","state":"draft","visibility":"public","pricingMode":"flat","pricePerPhoto":50.0,"description":null,"organizerName":null,"watermarkPolicy":null,"photoCount":0,"bannerUrl":null}"""
        fun error(code: String, message: String) = """{"success":false,"errors":[{"code":"$code","message":"$message"}]}"""
        fun verification(status: String) = """{"status":"$status","missing":[],"missingCodes":[],"suspendedAt":null,"suspensionReason":null}"""
        fun coupon(code: String, percent: Int) =
            """{"eventId":"ev-1","code":"$code","percentOff":$percent,"active":true,"usageLimit":10,"usageCount":0,"updatedAt":"2026-09-11T00:00:00Z"}"""
    }
}
