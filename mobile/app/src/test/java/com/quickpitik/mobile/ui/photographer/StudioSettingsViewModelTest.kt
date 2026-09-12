package com.quickpitik.mobile.ui.photographer

import android.app.Application
import android.net.Uri
import androidx.lifecycle.ViewModelStore
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.MAX_UPLOAD_BYTES
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
 * The settings tab's money and identity paths: which wallet a payout account
 * points at (QR required up front, required again when the number changes, an
 * account whose QR upload failed stays visible for a retry) and the settings
 * save chain (validate every image before the first PATCH, carry the hydrated
 * brand colour, stop at the first refused step with the backend's wording, PUT
 * the watermark placement only when it changed).
 *
 * These cases came from PhotographerDashboardViewModelMoneyTest with the
 * ViewModel split (plan 2 phase G). The `init` here loads four routes, so the
 * stubs below are the whole cold start — no events, earnings, verification,
 * inbox socket, USB or Room salvo to satisfy first.
 */
@RunWith(RobolectricTestRunner::class)
class StudioSettingsViewModelTest {
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
        route("GET $PAYOUTS") { ok("[]") }
        route("GET /api/v1/me/photographer/socials") { ok("[]") }
        route("GET /api/v1/regions") { ok("[]") }
    }

    @After fun tearDown() {
        viewModels.clear()
        SessionManager.getInstance(application).clearSession()
    }

    private fun route(key: String, respond: (RecordedRequest) -> MockResponse) { routes[key] = respond }
    private fun model() = StudioSettingsViewModel(application).also { viewModels.put("settings", it) }
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
        assertEquals(ActionState.Failed("Upload a payout QR code before saving."), vm.actionState.value)
        assertEquals(0, count("POST", PAYOUTS))

        route("POST $PAYOUTS") { ok(account("pa-1", qr = false)) }
        route("POST $PAYOUTS/pa-1/qr") { MockResponse().setResponseCode(429).setBody(error("RATE_LIMITED", "Slow down.")) }
        vm.addPayoutAccount("gcash", "Juan", "09171234567", picked("qr.png", 16))
        vm.actionState.awaitFirst { it.message?.startsWith("Payout setup is incomplete") == true }
        awaitUntil { count("GET", PAYOUTS) == 2 } // the startup load plus the refetch that shows the incomplete account
        assertEquals(1, count("POST", PAYOUTS))

        route("POST $PAYOUTS/pa-1/qr") { ok(account("pa-1", qr = true)) }
        vm.addPayoutAccount("gcash", "Juan", "09171234567", picked("qr.png", 16))
        vm.actionState.awaitFirst { it == ActionState.Done("Payout added.") }
    }

    @Test fun `updating a payout account requires a fresh QR when it has none or the number changed`() {
        route("GET $PAYOUTS") { ok("[${account("pa-1", qr = true)},${account("pa-2", qr = false, number = "09170000000")}]") }
        val vm = model()
        vm.payoutAccounts.awaitFirst { it.size == 2 }

        vm.updatePayoutAccount("pa-1", "Juan", "09179999999", qrUri = null) // number changed
        assertEquals(
            ActionState.Failed("Upload a QR code matching this payout account before saving."),
            vm.actionState.value,
        )
        vm.updatePayoutAccount("pa-2", "Juan", "09170000000", qrUri = null) // no QR on file
        assertEquals(
            ActionState.Failed("Upload a QR code matching this payout account before saving."),
            vm.actionState.value,
        )
        assertEquals(0, count("PATCH", "$PAYOUTS/pa-1") + count("PATCH", "$PAYOUTS/pa-2"))

        route("PATCH $PAYOUTS/pa-1") { ok(account("pa-1", qr = true)) }
        vm.updatePayoutAccount("pa-1", "Juan Dela Cruz", "09171234567", qrUri = null) // same number, QR on file
        vm.actionState.awaitFirst { it == ActionState.Done("Payout updated.") }
        assertTrue(bodies("PATCH", "$PAYOUTS/pa-1").single().contains(""""accountName":"Juan Dela Cruz""""))
    }

    @Test fun `settings save validates every picked image before the first PATCH`() {
        val vm = model()
        vm.brandSettings.awaitFirst { it != null }
        vm.saveSettings(
            brandName = "Juan Photos", bio = "", gcashName = "", gcashNumber = "", handle = "", regionCode = "",
            provinceCode = "", socialUrl = "", avatarUri = picked("avatar.jpg", MAX_UPLOAD_BYTES + 1), coverUri = null, watermarkUri = null,
        )
        assertEquals(
            ActionState.Failed("Avatar must be a readable image up to 8 MB."),
            vm.actionState.awaitFirst { it is ActionState.Failed },
        )
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
        val saved = CopyOnWriteArrayList<Unit>()

        vm.saveSettings(
            brandName = "Juan Photos", bio = "bio", gcashName = "", gcashNumber = "", handle = "juan", regionCode = "07",
            provinceCode = "0722", socialUrl = "", avatarUri = null, coverUri = null, watermarkUri = null,
            onSaved = { saved += Unit },
        )
        assertEquals(
            ActionState.Failed("That handle is already taken."),
            vm.actionState.awaitFirst { it is ActionState.Failed },
        )
        val brand = bodies("PUT", "/api/v1/me/photographer/brand").single()
        assertTrue(brand, brand.contains(""""brandColor":"amber"""") && brand.contains(""""brandName":"Juan Photos""""))
        assertEquals(1, count("PUT", "/api/v1/me/photographer/handle"))
        assertEquals(0, count("PUT", "/api/v1/me/photographer/region")) // the chain stopped at the handle
        assertEquals(0, saved.size) // a refused chain never reports a save
        vm.isSavingSettings.awaitFirst { !it }

        route("PUT /api/v1/me/photographer/handle") { ok("{}") }
        route("PUT /api/v1/me/photographer/region") { ok("{}") }
        route("PUT /api/v1/me/photographer/watermark/settings") { ok("{}") }
        vm.saveSettings(
            brandName = "Juan Photos", bio = "bio", gcashName = "", gcashNumber = "", handle = "juan", regionCode = "07",
            provinceCode = "0722", socialUrl = "", avatarUri = null, coverUri = null, watermarkUri = null,
            brandColor = "fresh", watermarkPlacement = WatermarkSettingsDto(), // equals the loaded placement
            onSaved = { saved += Unit },
        )
        vm.actionState.awaitFirst { it == ActionState.Done("Settings updated successfully!") }
        assertTrue(bodies("PUT", "/api/v1/me/photographer/brand").last().contains(""""brandColor":"fresh""""))
        assertEquals(1, count("PUT", "/api/v1/me/photographer/region"))
        assertEquals(0, count("PUT", "/api/v1/me/photographer/watermark/settings"))
        // A landed chain tells its caller, which is how the settings screen
        // refreshes the verification checklist a saved handle/region completes.
        awaitUntil { saved.size == 1 }

        vm.saveSettings(
            brandName = "Juan Photos", bio = "bio", gcashName = "", gcashNumber = "", handle = "", regionCode = "",
            provinceCode = "", socialUrl = "", avatarUri = null, coverUri = null, watermarkUri = null,
            watermarkPlacement = WatermarkSettingsDto(opacity = 0.5),
        )
        vm.actionState.awaitFirst { it == ActionState.Done("Settings updated successfully!") }
        assertEquals(1, count("PUT", "/api/v1/me/photographer/watermark/settings"))
    }

    private companion object {
        const val PAYOUTS = "/api/v1/me/photographer/payouts"
        const val BRAND = """{"brandName":"Juan Photos","brandColor":"amber","bio":"","handle":"juan","regionCode":"07","provinceCode":"0722","coverUrl":null,"watermarkUrl":null,"avatarUrl":null,"watermarkSettings":{"opacity":0.7,"scale":0.15,"landscapeX":1.0,"landscapeY":1.0,"portraitX":1.0,"portraitY":1.0}}"""
        fun error(code: String, message: String) = """{"success":false,"errors":[{"code":"$code","message":"$message"}]}"""
        fun account(id: String, qr: Boolean, number: String = "09171234567") =
            """{"id":"$id","method":"gcash","accountNumber":"$number","accountName":"Juan","qr":${if (qr) """{"dataUrl":"data:image/png;base64,AA==","uploadedAt":"2026-09-11T00:00:00Z"}""" else "null"},"isPrimary":true}"""
    }
}
