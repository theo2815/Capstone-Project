package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
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
import org.junit.Ignore
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.TimeUnit

/**
 * The money rules of the cart: what the coupon preview does with an empty
 * cart, a blank code and a refused code; what checkout actually transmits;
 * which orders count as already owned; and that removing a photo re-prices
 * the discount so the sheet's total never drifts from the charged one.
 */
@RunWith(RobolectricTestRunner::class)
class CartViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private val seen = CopyOnWriteArrayList<Pair<String, String>>() // path to body
    @Volatile private var cart = CART
    @Volatile private var orders: MockResponse = ok(ORDERS)

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("RUNNER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val path = request.path.orEmpty().substringBefore('?')
                val body = request.body.readUtf8()
                seen += path to body
                return when {
                    path == "/api/v1/me/cart" && request.method == "GET" -> ok(cart)
                    path.startsWith("/api/v1/me/cart/items/") -> ok("""{"removed":true}""")
                    path == "/api/v1/coupons/preview" -> preview(body)
                    path == "/api/v1/orders" -> ok(ORDER)
                    path == "/api/v1/me/orders" -> orders
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }
    }

    @After fun tearDown() = SessionManager.getInstance(application).clearSession()

    // Prices 12.50 off per photo in the request; "BAD" is refused the way the backend would.
    private fun preview(body: String): MockResponse {
        val code = Regex(""""code":"([^"]+)"""").find(body)?.groupValues?.get(1)
        if (code == "BAD") {
            return MockResponse().setResponseCode(400).setBody(
                """{"success":false,"errors":[{"code":"COUPON_INVALID","message":"That code isn't valid."}]}""",
            )
        }
        val count = Regex(""""photoIds":\[([^\]]*)\]""").find(body)!!.groupValues[1].split(',').count { it.isNotBlank() }
        val codeJson = code?.let { "\"$it\"" } ?: "null"
        return ok("""{"code":$codeJson,"percentOff":10,"items":[],"eligibleCount":$count,"discountTotal":${12.5 * count}}""")
    }

    private fun previews() = seen.filter { it.first == "/api/v1/coupons/preview" }.map { it.second }
    private fun orderBodies() = seen.filter { it.first == "/api/v1/orders" }.map { it.second }

    private fun newVm() = CartViewModel(application).also { vm ->
        if (cart != "[]") vm.cartItems.awaitFirst { it.isNotEmpty() }
    }

    private fun CartViewModel.previewAndWait(code: String?) {
        previewCoupon(code)
        couponLoading.awaitFirst { !it }
    }

    @Test fun `preview with an empty cart clears without a request, blank means auto, a refused code keeps the last preview`() {
        cart = "[]"
        val empty = newVm()
        empty.previewAndWait("SAVE10")
        assertNull(empty.couponPreview.value)
        assertTrue(previews().isEmpty())

        cart = CART
        val vm = newVm()
        vm.previewAndWait("   ")
        assertFalse(previews().single().contains("\"code\""))
        assertEquals(25.0, vm.couponPreview.value!!.discountTotal, 0.0)
        assertNull(vm.couponPreview.value!!.code)

        vm.previewAndWait("BAD")
        assertEquals("That code isn't valid.", vm.couponError.value)
        assertEquals(25.0, vm.couponPreview.value!!.discountTotal, 0.0)
    }

    @Test fun `checkout sends only a typed override and falls back to the session email`() {
        val vm = newVm()
        vm.previewAndWait("SAVE10")
        vm.checkout(null, "GCASH")
        vm.checkoutState.awaitFirst { it is CheckoutState.Success }
        val typed = orderBodies().single()
        assertTrue(typed, typed.contains(""""couponCode":"SAVE10"""") && typed.contains(""""recipientEmail":"juan@x.com""""))

        vm.resetCheckoutState()
        vm.previewAndWait(null)
        vm.checkout("other@x.com", "MAYA")
        vm.checkoutState.awaitFirst { it is CheckoutState.Success }
        val auto = orderBodies().last()
        assertFalse(auto, auto.contains("couponCode"))
        assertTrue(auto, auto.contains(""""recipientEmail":"other@x.com"""") && auto.contains(""""paymentMethod":"MAYA""""))
    }

    @Test fun `owned photo ids come from PAID and FULFILLED orders only, and are empty when orders fail`() {
        val vm = newVm()
        vm.fetchOrders()
        assertEquals(setOf("p1", "p2", "p6"), vm.ownedPhotoIds.awaitFirst { it.isNotEmpty() })

        orders = MockResponse().setResponseCode(500)
        val failed = newVm()
        failed.fetchOrders()
        failed.ordersState.awaitFirst { it is OrdersState.Error }
        assertEquals(emptySet<String>(), failed.ownedPhotoIds.awaitFirst { true })
    }

    @Test fun `removing a photo re-prices the coupon and an emptied cart drops the preview`() {
        val vm = newVm()
        vm.previewAndWait("SAVE10")
        assertEquals(25.0, vm.couponPreview.value!!.discountTotal, 0.0)

        vm.removeFromCart("p1")
        vm.couponPreview.awaitFirst { it?.discountTotal == 12.5 }
        val repriced = previews().last()
        assertTrue(repriced, repriced.contains(""""photoIds":["p2"]""") && repriced.contains(""""code":"SAVE10""""))

        vm.removeFromCart("p2")
        vm.couponPreview.awaitFirst { it == null }
        assertEquals(2, previews().size)
    }

    // Signs the runner out and a different runner in on the same phone, the way
    // MainActivity's sign-out and the login screen do.
    private fun switchAccount() = SessionManager.getInstance(application).run {
        clearSession()
        saveSession("t2", "RUNNER", "Maria", "maria@x.com")
    }

    // Security audit 2026-10-14, MOB-3.
    @Ignore("MS-1 reproduction of MOB-3; MS-2 removes this line")
    @Test fun `a change of account drops the previous runner's orders, even offline`() {
        val vm = newVm()
        vm.fetchOrders()
        assertEquals(setOf("p1", "p2", "p6"), vm.ownedPhotoIds.awaitFirst { it.isNotEmpty() })

        switchAccount()
        vm.ordersState.awaitFirst(1_000) { it !is OrdersState.Success }
        assertEquals(emptySet<String>(), vm.ownedPhotoIds.awaitFirst(1_000) { it.isEmpty() })

        // The next runner goes offline before opening Profile or Orders.
        orders = MockResponse().setResponseCode(500)
        vm.fetchOrders()
        vm.ordersState.awaitFirst(1_000) { it is OrdersState.Error }
    }

    @Ignore("MS-1 reproduction of MOB-3; MS-2 removes this line")
    @Test fun `an orders answer that lands after the account changed is dropped`() {
        orders = ok(ORDERS).setHeadersDelay(300, TimeUnit.MILLISECONDS)
        val vm = newVm()
        val inFlight = vm.fetchOrders()
        // Switch only once the request is on the server, so it is the answer that lands late.
        generateSequence { backend.server.takeRequest(5, TimeUnit.SECONDS)!! }.first { it.path!!.startsWith("/api/v1/me/orders") }

        switchAccount()
        runBlocking { inFlight.join() }

        assertFalse(vm.ordersState.value is OrdersState.Success)
    }

    private companion object {
        const val CART = """[{"photoId":"p1","eventId":"e1","thumbnailUrl":null,"price":100.0,"bib":"B-1","eventName":"Run","eventSlug":"run","tone":null,"time":null},{"photoId":"p2","eventId":"e1","thumbnailUrl":null,"price":100.0,"bib":null,"eventName":"Run","eventSlug":"run","tone":null,"time":null}]"""
        const val ORDER = """{"id":"o1","status":"PENDING","items":[{"photoId":"p1","price":100.0}],"totalAmount":200.0,"paymentMethod":"GCASH","createdAt":"2026-09-11T00:00:00Z","redirectUrl":"https://pay.example/x"}"""
        const val ORDERS = """{"items":[
            {"id":"o1","eventId":"e1","photoIds":["p1"],"total":100.0,"paymentMethod":"GCASH","paidAt":null,"eventName":null,"eventSlug":null,"eventDate":null,"eventState":null,"status":"PAID"},
            {"id":"o2","eventId":"e1","photoIds":["p2"],"total":100.0,"paymentMethod":"GCASH","paidAt":null,"eventName":null,"eventSlug":null,"eventDate":null,"eventState":null,"status":"FULFILLED"},
            {"id":"o3","eventId":"e1","photoIds":["p3"],"total":100.0,"paymentMethod":"GCASH","paidAt":null,"eventName":null,"eventSlug":null,"eventDate":null,"eventState":null,"status":"PENDING"},
            {"id":"o4","eventId":"e1","photoIds":["p4"],"total":100.0,"paymentMethod":"GCASH","paidAt":null,"eventName":null,"eventSlug":null,"eventDate":null,"eventState":null,"status":"REFUNDED"},
            {"id":"o5","eventId":"e1","photoIds":["p5"],"total":100.0,"paymentMethod":"GCASH","paidAt":null,"eventName":null,"eventSlug":null,"eventDate":null,"eventState":null,"status":null},
            {"id":"o6","eventId":"e1","photoIds":["p6"],"total":100.0,"paymentMethod":"GCASH","paidAt":null,"eventName":null,"eventSlug":null,"eventDate":null,"eventState":null,"status":"fulfilled"}
        ],"total":6,"offset":0,"limit":200}"""
    }
}
