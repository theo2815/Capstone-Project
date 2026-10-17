package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.SemanticsNodeInteraction
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.performSemanticsAction
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
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * The checkout sheet offers QR Ph only (owner decision 2026-10-17) and walks
 * the website's QR step: generate the code, then I've paid / Save / Cancel.
 */
@RunWith(RobolectricTestRunner::class)
class CheckoutSheetTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()
    private lateinit var application: Application

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("RUNNER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val path = request.path.orEmpty().substringBefore('?')
                return when {
                    path == "/api/v1/me/cart" -> ok(CART)
                    path == "/api/v1/coupons/preview" -> ok("""{"code":null,"percentOff":0,"items":[],"eligibleCount":0,"discountTotal":0.0}""")
                    path == "/api/v1/me/orders" -> ok("""{"items":[],"total":0,"offset":0,"limit":200}""")
                    path == "/api/v1/orders" -> ok(QR_ORDER)
                    path == "/api/v1/me/orders/o1/status" -> ok("""{"id":"o1","status":"PENDING","paidAt":null}""")
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }
    }

    @After fun tearDown() = SessionManager.getInstance(application).clearSession()

    @OptIn(ExperimentalTestApi::class)
    @Test fun `only QR Ph is offered, and generating a code shows the QR step`() {
        val vm = CartViewModel(application)
        vm.cartItems.awaitFirst { it.isNotEmpty() }
        runComposeUiTest {
            setContent { CheckoutSheet(cartViewModel = vm, onDismiss = {}) }
            waitUntil(timeoutMillis = 5_000) { onAllNodesWithText("Review & pay").fetchSemanticsNodes().isNotEmpty() }
            onNodeWithText("QR Ph").assertExists()
            listOf("GCash e-wallet", "Maya e-wallet", "Credit / debit card").forEach {
                onNodeWithText(it).assertDoesNotExist()
            }

            onNodeWithText("Generate QR to pay", substring = true).assertIsEnabled().tap()
            waitUntil(timeoutMillis = 5_000) { onAllNodesWithText("Scan to pay.").fetchSemanticsNodes().isNotEmpty() }
            onNodeWithText("Amount due · ₱250.00", ignoreCase = true).assertExists()
            onNodeWithText("Save QR code").assertExists()
            onNodeWithText("I've paid").assertExists()

            onNodeWithText("Cancel payment").tap()
            waitUntil(timeoutMillis = 5_000) { onAllNodesWithText("Cancel this payment?").fetchSemanticsNodes().isNotEmpty() }
            onNodeWithText("Keep waiting").tap()
            waitUntil(timeoutMillis = 5_000) { onAllNodesWithText("Cancel this payment?").fetchSemanticsNodes().isEmpty() }
        }
        assertTrue(vm.checkoutState.value is CheckoutState.AwaitingQr)
    }

    @Test fun `pure QR helpers`() {
        assertEquals(QrOutcome.EXPIRED, qrEndOutcome(expiresAtMillis = 10_000, now = 6_000))
        assertEquals(QrOutcome.FAILED, qrEndOutcome(expiresAtMillis = 10_000, now = 4_999))
        assertEquals(QrOutcome.EXPIRED, qrEndOutcome(expiresAtMillis = null, now = 0))
        assertEquals(1_000L, parseInstantMillis("1970-01-01T08:00:01+08:00"))
        assertNull(parseInstantMillis("soon"))

        assertEquals(listOf(0x89.toByte(), 'P'.code.toByte()), qrImageBytes("data:image/png;base64,iVA=")!!.toList())
        assertNull(qrImageBytes("https://example.com/qr.png"))
        assertNull(qrImageBytes("data:text/plain;base64,aGk="))
        assertNull(qrImageBytes("data:image/svg+xml;base64,aGk="))
        assertNull(qrImageBytes("data:image/png;base64," + "A".repeat(600 * 1024)))

        assertEquals(QrWaitMode.AWAITING, qrWaitMode(paidClaimedAt = null, now = 0))
        assertEquals(QrWaitMode.CONFIRMING, qrWaitMode(paidClaimedAt = 0, now = 59_999))
        assertEquals(QrWaitMode.SLOW, qrWaitMode(paidClaimedAt = 0, now = 60_000))
        assertEquals("1:05", formatElapsed(65_400))
        assertEquals("0:00", formatElapsed(-5))
    }

    // A semantics click: the sheet's scroll area can sit outside Robolectric's
    // window, where an injected touch would miss.
    private fun SemanticsNodeInteraction.tap() = performSemanticsAction(SemanticsActions.OnClick)

    private companion object {
        const val CART = """[{"photoId":"p1","eventId":"e1","thumbnailUrl":null,"price":125.0,"bib":"B-1","eventName":"Run","eventSlug":"run","tone":null,"time":null},{"photoId":"p2","eventId":"e1","thumbnailUrl":null,"price":125.0,"bib":null,"eventName":"Run","eventSlug":"run","tone":null,"time":null}]"""
        const val QR_ORDER = """{"id":"o1","status":"PENDING","items":[{"photoId":"p1","price":125.0}],"totalAmount":250.0,"checkoutTotalAmount":250.0,"paymentMethod":"qrph","createdAt":"2026-09-11T00:00:00Z","qrPh":{"imageUrl":"data:image/png;base64,iVBORw0KGgo=","expiresAt":"2099-01-01T00:00:00Z","returnToken":null,"testUrl":null}}"""
    }
}
