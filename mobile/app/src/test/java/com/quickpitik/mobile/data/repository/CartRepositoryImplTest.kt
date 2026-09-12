package com.quickpitik.mobile.data.repository

import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.ok
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.CopyOnWriteArrayList

/**
 * The checkout idempotency contract. The Idempotency-Key header is the
 * backend's dedupe handle: a re-tap after a timeout must RESEND the same key
 * (a fresh UUID per call once minted a second order), a genuinely different
 * order rotates it, and a minted order retires it. Plus the two guards around
 * it: an empty cart never reaches the network, and a 429 reads as the
 * backend's own copy rather than "HTTP 429".
 */
@RunWith(RobolectricTestRunner::class)
class CartRepositoryImplTest {
    @get:Rule val backend = MockBackend()

    private val requests = CopyOnWriteArrayList<RecordedRequest>()
    @Volatile private var order: MockResponse = MockResponse().setResponseCode(500)
    @Volatile private var addItem: MockResponse = MockResponse().setResponseCode(404)
    @Volatile private var removeItem: MockResponse = ok("""{"removed":true}""")
    private val repo = CartRepositoryImpl()
    private val session = SessionManager.getInstance(ApplicationProvider.getApplicationContext())

    @Before fun routes() {
        // The real auth interceptor refuses a bearer no session issued.
        session.saveSession("t", "RUNNER", "Juan", "juan@x.com")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                requests += request
                return when (request.path) {
                    "/api/v1/me/cart" -> ok(CART)
                    "/api/v1/me/cart/items" -> addItem
                    "/api/v1/me/cart/items/p1" -> removeItem
                    "/api/v1/orders" -> order
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }
    }

    @After fun signOut() = session.clearSession()

    private fun orderKeys() =
        requests.filter { it.path == "/api/v1/orders" }.map { it.getHeader("Idempotency-Key")!! }

    private fun checkout(method: String = "GCASH", email: String? = "juan@x.com", coupon: String? = null) =
        runBlocking { repo.checkout("t", email, method, coupon) }

    @Test fun `a retry resends the same key, a different order rotates it, a minted order retires it`() = runBlocking {
        repo.fetchCart("t").getOrThrow()
        assertTrue(checkout().isFailure)
        assertTrue(checkout().isFailure)
        val keys = orderKeys()
        assertEquals(2, keys.size)
        assertEquals(keys[0], keys[1])

        checkout(method = "MAYA")
        checkout(email = "other@x.com")
        checkout(coupon = "SAVE10")
        repo.removeFromCart("t", "p1").getOrThrow()
        checkout()
        val rotated = orderKeys().drop(2)
        assertEquals(4, rotated.size)
        assertEquals(5, (rotated + keys[0]).toSet().size)

        order = ok(ORDER)
        assertTrue(checkout().isSuccess)
        order = MockResponse().setResponseCode(500)
        assertTrue(checkout().isFailure)
        val tail = orderKeys().takeLast(3)
        assertEquals(tail[0], tail[1]) // the successful call reused the pending key
        assertNotEquals(tail[1], tail[2]) // a minted order retires it
    }

    @Test fun `an empty cart never reaches the network and a 429 reads as the backend's copy`() = runBlocking {
        assertEquals("Cart is empty", checkout().exceptionOrNull()?.message)
        assertTrue(requests.none { it.path == "/api/v1/orders" })

        repo.fetchCart("t").getOrThrow()
        order = MockResponse().setResponseCode(429).setBody(
            """{"success":false,"errors":[{"code":"RATE_LIMITED","message":"Slow down. Try again in a minute."}]}""",
        )
        assertEquals("Slow down. Try again in a minute.", checkout(coupon = "SAVE10").exceptionOrNull()?.message)
        val body = requests.last { it.path == "/api/v1/orders" }.body.readUtf8()
        assertTrue(body, body.contains(""""couponCode":"SAVE10"""") && body.contains(""""clientPlatform":"android""""))
    }

    // A 200 the backend refuses inside the envelope. That is the branch the
    // optimistic revert lives on now that both sites read the envelope through
    // toResult; a THROWN failure takes the catch beside it, which reverts on
    // its own and is the path the older tests exercise.
    private fun refused(code: String, message: String) = MockResponse()
        .setBody("""{"success":false,"errors":[{"code":"$code","message":"$message"}]}""")

    private fun photo(id: String) = PhotoDto(
        id = id, bib = null, km = null, time = "00:00", tone = 0, span = "portrait",
        price = 100.0, imageUrl = null, alt = null,
    )

    @Test fun `a cart add or remove the envelope refuses puts the optimistic row back`() = runBlocking {
        repo.fetchCart("t").getOrThrow()
        val before = repo.cartItems.value.map { it.photoId }
        assertEquals(listOf("p1", "p2"), before)

        addItem = refused("PHOTO_UNAVAILABLE", "That photo is no longer for sale.")
        val add = repo.addToCart("t", photo("p3"), "e1", "run", "Run")
        assertEquals("That photo is no longer for sale.", add.exceptionOrNull()?.message)
        assertEquals(before, repo.cartItems.value.map { it.photoId })

        removeItem = refused("CART_ITEM_MISSING", "That item is already gone.")
        val remove = repo.removeFromCart("t", "p1")
        assertEquals("That item is already gone.", remove.exceptionOrNull()?.message)
        assertEquals(before, repo.cartItems.value.map { it.photoId })
    }

    private companion object {
        const val CART = """[{"photoId":"p1","eventId":"e1","thumbnailUrl":null,"price":100.0,"bib":"B-1","eventName":"Run","eventSlug":"run","tone":null,"time":null},{"photoId":"p2","eventId":"e1","thumbnailUrl":null,"price":100.0,"bib":null,"eventName":"Run","eventSlug":"run","tone":null,"time":null}]"""
        const val ORDER = """{"id":"o1","status":"PENDING","items":[{"photoId":"p2","price":100.0}],"totalAmount":100.0,"paymentMethod":"GCASH","createdAt":"2026-09-11T00:00:00Z","redirectUrl":"https://pay.example/x"}"""
    }
}
