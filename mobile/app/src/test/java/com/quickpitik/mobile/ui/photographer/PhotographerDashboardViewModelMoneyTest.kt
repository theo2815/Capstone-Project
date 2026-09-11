package com.quickpitik.mobile.ui.photographer

import android.app.Application
import android.net.Uri
import androidx.lifecycle.ViewModelStore
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.MAX_UPLOAD_BYTES
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.WatermarkSettingsDto
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
import org.robolectric.Shadows
import java.io.ByteArrayInputStream
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList

/**
 * The photographer's money paths: which wallet a payout account points at
 * (QR required up front, required again when the number changes, an account
 * whose QR upload failed stays visible for a retry), the payout request's
 * backend reasons, the settings save chain (validate every image before the
 * first PATCH, carry the hydrated brand colour, stop at the first refused
 * step with the backend's wording, PUT the watermark placement only when it
 * changed), coupons, and the fail-open approval railguard. Same construction
 * as PhotoManagementViewModelTest: the real ViewModel over a MockWebServer.
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
        route("GET /api/v1/me/photographer/brand") { ok(BRAND) }
        route("GET /api/v1/me/photographer/payouts") { ok("[]") }
        route("GET /api/v1/me/photographer/verification") { ok(verification("approved")) }
    }

    @After fun tearDown() {
        viewModels.clear()
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

    // A picked image: a fresh stream every call, because the resolver hands back the same one each time.
    private fun picked(name: String, bytes: Int): String {
        val uri = Uri.parse("content://test/$name")
        Shadows.shadowOf(application.contentResolver).registerInputStream(uri, ByteArrayInputStream(ByteArray(bytes) { 1 }))
        return uri.toString()
    }

    @Test fun `adding a payout account needs a QR up front, keeps an account whose QR failed, and reports success otherwise`() {
        val vm = model()
        vm.addPayoutAccount("gcash", "Juan", "09171234567", qrUri = null)
        assertEquals("Error: Upload a payout QR code before saving.", vm.settingsActionState.value)
        assertEquals(0, count("POST", PAYOUTS))

        route("POST $PAYOUTS") { ok(account("pa-1", qr = false)) }
        route("POST $PAYOUTS/pa-1/qr") { MockResponse().setResponseCode(429).setBody(error("RATE_LIMITED", "Slow down.")) }
        vm.addPayoutAccount("gcash", "Juan", "09171234567", picked("qr.png", 16))
        vm.settingsActionState.awaitFirst { it != null && it.startsWith("Error: Payout setup is incomplete") }
        awaitUntil { count("GET", PAYOUTS) == 2 } // the startup load plus the refetch that shows the incomplete account
        assertEquals(1, count("POST", PAYOUTS))

        route("POST $PAYOUTS/pa-1/qr") { ok(account("pa-1", qr = true)) }
        vm.addPayoutAccount("gcash", "Juan", "09171234567", picked("qr.png", 16))
        vm.settingsActionState.awaitFirst { it == "Success: Payout added." }
    }

    @Test fun `updating a payout account requires a fresh QR when it has none or the number changed`() {
        route("GET $PAYOUTS") { ok("[${account("pa-1", qr = true)},${account("pa-2", qr = false, number = "09170000000")}]") }
        val vm = model()
        vm.payoutAccounts.awaitFirst { it.size == 2 }

        vm.updatePayoutAccount("pa-1", "Juan", "09179999999", qrUri = null) // number changed
        assertEquals("Error: Upload a QR code matching this payout account before saving.", vm.settingsActionState.value)
        vm.updatePayoutAccount("pa-2", "Juan", "09170000000", qrUri = null) // no QR on file
        assertEquals("Error: Upload a QR code matching this payout account before saving.", vm.settingsActionState.value)
        assertEquals(0, count("PATCH", "$PAYOUTS/pa-1") + count("PATCH", "$PAYOUTS/pa-2"))

        route("PATCH $PAYOUTS/pa-1") { ok(account("pa-1", qr = true)) }
        vm.updatePayoutAccount("pa-1", "Juan Dela Cruz", "09171234567", qrUri = null) // same number, QR on file
        vm.settingsActionState.awaitFirst { it == "Success: Payout updated." }
        assertTrue(bodies("PATCH", "$PAYOUTS/pa-1").single().contains(""""accountName":"Juan Dela Cruz""""))
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

    @Test fun `settings save validates every picked image before the first PATCH`() {
        val vm = model()
        vm.brandSettings.awaitFirst { it != null }
        vm.saveSettings(
            brandName = "Juan Photos", bio = "", gcashName = "", gcashNumber = "", handle = "", regionCode = "",
            provinceCode = "", socialUrl = "", avatarUri = picked("avatar.jpg", MAX_UPLOAD_BYTES + 1), coverUri = null, watermarkUri = null,
        )
        assertEquals("Error: Avatar must be a readable image up to 8 MB.", vm.settingsActionState.awaitFirst { it != null && it.startsWith("Error") })
        assertEquals(0, count("PUT", "/api/v1/me/photographer/brand"))
        vm.isSavingSettings.awaitFirst { !it }
    }

    @Test fun `settings save carries the hydrated brand colour, stops at the first refused step, and PUTs only a changed placement`() {
        route("PUT /api/v1/me/photographer/brand") { ok("{}") }
        route("PUT /api/v1/me/photographer/handle") {
            MockResponse().setResponseCode(409).setBody(error("HANDLE_TAKEN", "That handle is already taken."))
        }
        val vm = model()
        vm.brandSettings.awaitFirst { it?.brandColor == "amber" }

        vm.saveSettings(
            brandName = "Juan Photos", bio = "bio", gcashName = "", gcashNumber = "", handle = "juan", regionCode = "07",
            provinceCode = "0722", socialUrl = "", avatarUri = null, coverUri = null, watermarkUri = null,
        )
        assertEquals("Error: That handle is already taken.", vm.settingsActionState.awaitFirst { it != null && it.startsWith("Error") })
        val brand = bodies("PUT", "/api/v1/me/photographer/brand").single()
        assertTrue(brand, brand.contains(""""brandColor":"amber"""") && brand.contains(""""brandName":"Juan Photos""""))
        assertEquals(1, count("PUT", "/api/v1/me/photographer/handle"))
        assertEquals(0, count("PUT", "/api/v1/me/photographer/region")) // the chain stopped at the handle
        vm.isSavingSettings.awaitFirst { !it }

        route("PUT /api/v1/me/photographer/handle") { ok("{}") }
        route("PUT /api/v1/me/photographer/region") { ok("{}") }
        route("PUT /api/v1/me/photographer/watermark/settings") { ok("{}") }
        vm.saveSettings(
            brandName = "Juan Photos", bio = "bio", gcashName = "", gcashNumber = "", handle = "juan", regionCode = "07",
            provinceCode = "0722", socialUrl = "", avatarUri = null, coverUri = null, watermarkUri = null,
            brandColor = "fresh", watermarkPlacement = WatermarkSettingsDto(), // equals the loaded placement
        )
        vm.settingsActionState.awaitFirst { it == "Success: Settings updated successfully!" }
        assertTrue(bodies("PUT", "/api/v1/me/photographer/brand").last().contains(""""brandColor":"fresh""""))
        assertEquals(1, count("PUT", "/api/v1/me/photographer/region"))
        assertEquals(0, count("PUT", "/api/v1/me/photographer/watermark/settings"))

        vm.saveSettings(
            brandName = "Juan Photos", bio = "bio", gcashName = "", gcashNumber = "", handle = "", regionCode = "",
            provinceCode = "", socialUrl = "", avatarUri = null, coverUri = null, watermarkUri = null,
            watermarkPlacement = WatermarkSettingsDto(opacity = 0.5),
        )
        vm.settingsActionState.awaitFirst { it == "Success: Settings updated successfully!" }
        assertEquals(1, count("PUT", "/api/v1/me/photographer/watermark/settings"))
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
        const val BRAND = """{"brandName":"Juan Photos","brandColor":"amber","bio":"","handle":"juan","regionCode":"07","provinceCode":"0722","coverUrl":null,"watermarkUrl":null,"avatarUrl":null,"watermarkSettings":{"opacity":0.7,"scale":0.15,"landscapeX":1.0,"landscapeY":1.0,"portraitX":1.0,"portraitY":1.0}}"""
        const val PAYOUT = """{"id":"py-1","weekOf":"2026-09-07","amount":500.0,"status":"pending","settledAt":null,"method":"gcash","reference":null,"holdReason":null}"""
        fun error(code: String, message: String) = """{"success":false,"errors":[{"code":"$code","message":"$message"}]}"""
        fun verification(status: String) = """{"status":"$status","missing":[],"missingCodes":[],"suspendedAt":null,"suspensionReason":null}"""
        fun account(id: String, qr: Boolean, number: String = "09171234567") =
            """{"id":"$id","method":"gcash","accountNumber":"$number","accountName":"Juan","qr":${if (qr) """{"dataUrl":"data:image/png;base64,AA==","uploadedAt":"2026-09-11T00:00:00Z"}""" else "null"},"isPrimary":true}"""
        fun coupon(code: String, percent: Int) =
            """{"eventId":"ev-1","code":"$code","percentOff":$percent,"active":true,"usageLimit":10,"usageCount":0,"updatedAt":"2026-09-11T00:00:00Z"}"""
    }
}
