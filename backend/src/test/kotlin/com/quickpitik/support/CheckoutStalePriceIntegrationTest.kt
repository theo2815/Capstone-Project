package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.orders.CreateOrderItem
import com.quickpitik.dto.orders.CreateOrderRequest
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponse
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseAttributes
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseEnvelope
import com.quickpitik.entity.*
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.*
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.orders.*
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** AF-12 (review R-4): only a real database can show the stored price survives the refused checkout. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class CheckoutStalePriceIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var carts: CartItemRepository
    @Autowired lateinit var checkout: OrderService
    @Autowired lateinit var cart: com.quickpitik.service.cart.CartService
    @Autowired lateinit var jdbc: JdbcTemplate
    @MockitoBean lateinit var storage: StorageService
    @MockitoBean lateinit var provider: AiApiClient
    @MockitoBean lateinit var paymongo: PaymongoClient
    private lateinit var event: Event
    private lateinit var photo: Photo
    private lateinit var buyer: User

    @BeforeEach fun fixture() {
        val owner = users.saveAndFlush(User(email = "stale-owner-${UUID.randomUUID()}@test.local", passwordHash = "unused",
            name = "Photographer", role = Role.PHOTOGRAPHER))
        buyer = users.saveAndFlush(User(email = "stale-buyer-${UUID.randomUUID()}@test.local", passwordHash = "unused",
            name = "Buyer", role = Role.RUNNER))
        event = events.saveAndFlush(Event(slug = "stale-${UUID.randomUUID()}", name = "Cebu run",
            date = LocalDate.of(2026, 10, 5), location = "Cebu", status = EventStatus.ACTIVE, photoCount = 1))
        photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
            s3Key = "synthetic/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal("125.00"), phash = UUID.randomUUID().leastSignificantBits,
            uploadedAt = OffsetDateTime.now().minusMinutes(5)))
        Mockito.`when`(paymongo.createCheckoutSession(anyArg(), anyArg())).thenReturn(
            PaymongoCheckoutSessionResponse(PaymongoCheckoutSessionResponseEnvelope(id = "cs_${UUID.randomUUID()}",
                attributes = PaymongoCheckoutSessionResponseAttributes(checkoutUrl = "https://pay.test/checkout"))))
    }

    private fun <T> anyArg(): T = Mockito.any()

    private fun cartRow(userId: UUID, price: String) =
        jdbc.update("INSERT INTO cart_items (user_id, photo_id, event_id, price_php_at_add) VALUES (?, ?, ?, ?)",
            userId, photo.id, event.id, BigDecimal(price))

    private fun stored(userId: UUID): BigDecimal =
        jdbc.queryForObject("SELECT price_php_at_add FROM cart_items WHERE user_id = ? AND photo_id = ?",
            BigDecimal::class.java, userId, photo.id)!!

    private fun orderCount(userId: UUID) =
        jdbc.queryForObject("SELECT count(*) FROM orders WHERE user_id = ?", Int::class.java, userId)

    private fun pay() = checkout.create(buyer.id, CreateOrderRequest(items = listOf(CreateOrderItem(photo.id, event.id)),
        paymentMethod = "gcash"), UUID.randomUUID().toString())

    @Test fun `the refresh raises a lower stored price and leaves others alone`() {
        val other = users.saveAndFlush(User(email = "stale-other-${UUID.randomUUID()}@test.local", passwordHash = "unused",
            name = "Other", role = Role.RUNNER))
        cartRow(buyer.id, "100.00")
        cartRow(other.id, "100.00")

        assertEquals(1, carts.raisePriceAtAdd(buyer.id, photo.id, BigDecimal("125.00")))

        assertEquals(0, BigDecimal("125.00").compareTo(stored(buyer.id)))
        assertEquals(0, BigDecimal("100.00").compareTo(stored(other.id)))
        assertEquals(0, carts.raisePriceAtAdd(buyer.id, photo.id, BigDecimal("125.00")), "equal price is left alone")
        assertEquals(0, carts.raisePriceAtAdd(buyer.id, photo.id, BigDecimal("110.00")), "a lower live price never lowers it")
        assertEquals(0, BigDecimal("125.00").compareTo(stored(buyer.id)))
    }

    @Test fun `the lowering query lowers a higher stored price only, and only for that buyer`() {
        val other = users.saveAndFlush(User(email = "stale-other-${UUID.randomUUID()}@test.local", passwordHash = "unused",
            name = "Other", role = Role.RUNNER))
        cartRow(buyer.id, "125.00")
        cartRow(other.id, "125.00")

        assertEquals(1, carts.lowerPriceAtAdd(buyer.id, photo.id, BigDecimal("80.00")))

        assertEquals(0, BigDecimal("80.00").compareTo(stored(buyer.id)))
        assertEquals(0, BigDecimal("125.00").compareTo(stored(other.id)))
        assertEquals(0, carts.lowerPriceAtAdd(buyer.id, photo.id, BigDecimal("90.00")), "a higher live price never raises it")
        assertEquals(0, BigDecimal("80.00").compareTo(stored(buyer.id)))
    }

    // AF-18 F5: added at 125, the old app's cart showed 80, the price went back to 125.
    @Test fun `a drop shown by the cart read makes the rise back a refusal for an old app`() {
        cartRow(buyer.id, "125.00")
        Mockito.`when`(storage.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://cdn/x.jpg")
        jdbc.update("UPDATE photos SET price_php = 80.00 WHERE id = ?", photo.id)

        assertEquals(0, BigDecimal("80.00").compareTo(cart.list(buyer.id).single().price))
        assertEquals(0, BigDecimal("80.00").compareTo(stored(buyer.id)))

        jdbc.update("UPDATE photos SET price_php = 125.00 WHERE id = ?", photo.id)
        val error = assertFailsWith<ConflictException> { pay() }
        assertEquals("Photo price changed from ₱80.00 to ₱125.00. Refresh your cart and try again.", error.message)
        pay()
        assertEquals(1, orderCount(buyer.id))
    }

    @Test fun `after a refused checkout with no expected price the stored price is the live price`() {
        cartRow(buyer.id, "100.00")

        val error = assertFailsWith<ConflictException> { pay() }

        assertEquals(ErrorCodes.CART_ITEM_PRICE_CHANGED, error.code)
        assertEquals("Photo price changed from ₱100.00 to ₱125.00. Refresh your cart and try again.", error.message)
        assertEquals(0, orderCount(buyer.id))
        assertEquals(0, BigDecimal("125.00").compareTo(stored(buyer.id)))
        Mockito.verifyNoInteractions(paymongo)

        pay()
        assertEquals(1, orderCount(buyer.id))
    }
}
