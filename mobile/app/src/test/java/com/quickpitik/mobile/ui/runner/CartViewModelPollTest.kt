package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.ok
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Ignore
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger

/**
 * The /orders/return poll: every terminal branch, the 404 grace, the
 * verify-on-last-attempt rule, the guest endpoints, and that re-entering the
 * route cancels the previous loop instead of racing it. Runs on a
 * StandardTestDispatcher so the thirty 2 s delays pass in virtual time; the
 * short sleeps only wait for the local HTTP round trip.
 */
@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
class CartViewModelPollTest {
    @get:Rule val main = MainDispatcherRule(StandardTestDispatcher())
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private val session get() = SessionManager.getInstance(application)
    private val requests = CopyOnWriteArrayList<RecordedRequest>()
    private val polls = AtomicInteger()
    @Volatile private var script: List<Any> = emptyList() // a status string or an HTTP code per poll
    @Volatile private var cart = "[]"

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        session.clearSession()
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                requests += request
                val path = request.path.orEmpty()
                return when {
                    path.startsWith("/api/v1/me/orders/o1/status") || path.startsWith("/api/v1/orders/o1/status") ->
                        when (val step = script.getOrElse(polls.getAndIncrement()) { "PENDING" }) {
                            is Int -> MockResponse().setResponseCode(step)
                            else -> ok("""{"id":"o1","status":"$step"}""")
                        }
                    path == "/api/v1/me/cart" && request.method == "DELETE" -> ok("""{"cleared":0}""")
                    path == "/api/v1/me/cart" -> ok(cart)
                    path == "/api/v1/me/orders/o1" || path.startsWith("/api/v1/orders/o1?") -> ok(DETAIL)
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }
    }

    @After fun tearDown() = session.clearSession()

    private fun statusPaths() = requests.map { it.path.orEmpty() }.filter { it.contains("/status") }

    private fun signedIn(): CartViewModel {
        session.saveSession("t", "RUNNER", "Juan", "juan@x.com")
        return CartViewModel(application)
    }

    // Each turn of the scheduler runs whatever the last HTTP reply queued and
    // skips the poll's delay in virtual time, until the loop settles.
    private fun settle(vm: CartViewModel): OrderReturnState {
        val deadline = System.currentTimeMillis() + 10_000
        while (System.currentTimeMillis() < deadline) {
            main.scheduler.advanceUntilIdle()
            when (val state = vm.orderReturnState.value) {
                is OrderReturnState.Paid, is OrderReturnState.Failed, OrderReturnState.Timeout -> return state
                else -> Thread.sleep(5)
            }
        }
        throw AssertionError("poll never settled: ${vm.orderReturnState.value}")
    }

    private fun runUntil(done: () -> Boolean) {
        val deadline = System.currentTimeMillis() + 5_000
        while (!done() && System.currentTimeMillis() < deadline) {
            main.scheduler.advanceUntilIdle()
            Thread.sleep(5)
        }
        assertTrue("timed out", done())
    }

    private fun poll(vm: CartViewModel, shareToken: String? = null) = settle(vm.also { it.pollOrderReturn("o1", shareToken) })

    @Test fun `PAID clears the cart and hydrates the receipt exactly once`() {
        script = listOf("PENDING", "PAID")
        val paid = poll(signedIn()) as OrderReturnState.Paid
        assertEquals("o1", paid.order.id)
        assertEquals(listOf("/api/v1/me/orders/o1/status", "/api/v1/me/orders/o1/status"), statusPaths())
        assertEquals(1, requests.count { it.method == "DELETE" && it.path == "/api/v1/me/cart" })
        assertEquals(1, requests.count { it.path == "/api/v1/me/orders/o1" })
    }

    @Test fun `EXPIRED and REFUNDED are terminal failures with their own copy`() {
        script = listOf("EXPIRED")
        val expired = poll(signedIn()) as OrderReturnState.Failed
        assertTrue(expired.message, expired.message.contains("expired"))

        polls.set(0)
        script = listOf("REFUNDED")
        val refunded = poll(signedIn()) as OrderReturnState.Failed
        assertTrue(refunded.message, refunded.message.contains("refunded"))
        assertEquals(0, requests.count { it.method == "DELETE" })
    }

    @Test fun `a fresh order may 404 twice, a third 404 fails fast`() {
        script = listOf(404, 404, "PAID")
        assertTrue(poll(signedIn()) is OrderReturnState.Paid)

        polls.set(0)
        requests.clear()
        script = listOf(404, 404, 404)
        val failed = poll(signedIn()) as OrderReturnState.Failed
        assertTrue(failed.message, failed.message.contains("couldn't find"))
        assertEquals(3, statusPaths().size)
    }

    @Test fun `thirty pending polls end in Timeout and only the last asks PayMongo to verify`() {
        script = List(30) { "PENDING" }
        assertEquals(OrderReturnState.Timeout, poll(signedIn()))
        val paths = statusPaths()
        assertEquals(30, paths.size)
        assertTrue(paths.dropLast(1).none { it.contains("verify") })
        assertTrue(paths.last(), paths.last().endsWith("?verify=true"))
    }

    @Test fun `a guest return uses the token endpoints and never touches the account cart`() {
        script = listOf("PAID")
        assertTrue(poll(CartViewModel(application), shareToken = "tok") is OrderReturnState.Paid)
        assertEquals(listOf("/api/v1/orders/o1/status?token=tok"), statusPaths())
        assertEquals(1, requests.count { it.path == "/api/v1/orders/o1?token=tok" })
        assertEquals(0, requests.count { it.method == "DELETE" })

        val signedOut = CartViewModel(application)
        signedOut.pollOrderReturn("o1")
        assertTrue(signedOut.orderReturnState.value is OrderReturnState.Failed)
    }

    @Test fun `re-entering the receipt cancels the previous poll instead of racing it`() {
        script = listOf("PENDING", "PENDING", "PENDING", "PAID")
        val vm = signedIn()
        vm.pollOrderReturn("o1")
        val deadline = System.currentTimeMillis() + 5_000
        while (statusPaths().size < 2 && System.currentTimeMillis() < deadline) {
            main.scheduler.advanceUntilIdle()
            Thread.sleep(5)
        }
        assertEquals(2, statusPaths().size)

        assertTrue(poll(vm) is OrderReturnState.Paid)
        // 2 polls from the cancelled loop + 2 from the fresh one; a surviving
        // first loop would have kept polling past that.
        assertEquals(4, statusPaths().size)
    }

    // Security audit 2026-10-14, MOB-2: a receipt link can be anyone's order.
    @Ignore("MS-5 reproduction of MOB-2; MS-6 removes this line")
    @Test fun `a stranger's receipt link never touches the signed-in runner's server cart`() {
        cart = CART_TWO
        val vm = signedIn()
        runUntil { vm.cartItems.value.size == 2 }
        script = listOf("PAID")

        assertTrue(poll(vm, shareToken = "tok") is OrderReturnState.Paid)

        assertEquals(0, requests.count { it.method == "DELETE" })
    }

    private companion object {
        const val DETAIL = """{"id":"o1","eventId":"e1","photoIds":["p1"],"total":100.0,"paymentMethod":"GCASH","paidAt":null,"eventName":null,"eventSlug":null,"status":"PAID","photos":[]}"""
        const val CART_TWO = """[{"photoId":"p1","eventId":"e1","thumbnailUrl":null,"price":100.0,"bib":null,"eventName":"Run","eventSlug":"run","tone":null,"time":null},{"photoId":"p9","eventId":"e1","thumbnailUrl":null,"price":100.0,"bib":null,"eventName":"Run","eventSlug":"run","tone":null,"time":null}]"""
    }
}
