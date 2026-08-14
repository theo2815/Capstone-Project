package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.DownloadGrant
import com.quickpitik.entity.DownloadGrantId
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.entity.Photo
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Runner-flow audit (2026-05-27), "Orders + downloads" item O1.
//
// share_token authorizes the bundle download by top-level navigation, so it
// travels in email and can be forwarded. V27 bounds it at
// orders.token_expires_at; expiry surfaces as NOT_FOUND like every other
// token failure so probing by id still teaches an attacker nothing.
class OrderBundleServiceTest {

    private lateinit var orderRepository: OrderRepository
    private lateinit var orderItemRepository: OrderItemRepository
    private lateinit var photoRepository: PhotoRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var downloadGrantRepository: DownloadGrantRepository
    private lateinit var storageService: StorageService
    private lateinit var service: OrderBundleService

    private val token = "a".repeat(64)

    @BeforeEach
    fun setUp() {
        orderRepository = Mockito.mock(OrderRepository::class.java)
        orderItemRepository = Mockito.mock(OrderItemRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        downloadGrantRepository = Mockito.mock(DownloadGrantRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        service = OrderBundleService(
            orderRepository,
            orderItemRepository,
            photoRepository,
            eventRepository,
            downloadGrantRepository,
            storageService,
        )
    }

    @Test
    fun `an expired share token is refused`() {
        val order = order(tokenExpiresAt = OffsetDateTime.now().minusDays(1))
        stub(order)

        val ex = assertFailsWith<NotFoundException> { service.prepare(order.id, token) }

        assertEquals(ErrorCodes.ORDER_NOT_FOUND, ex.code)
    }

    @Test
    fun `a token inside its window still resolves the bundle`() {
        val order = order(tokenExpiresAt = OffsetDateTime.now().plusDays(30))
        stub(order)

        val spec = service.prepare(order.id, token)

        assertEquals(1, spec.entries.size)
        // Single-photo orders collapse to the raw image rather than a ZIP.
        assertEquals("image/jpeg", spec.contentType)
    }

    @Test
    fun `expiry is checked, not merely stored`() {
        // Guards against the check being wired to the wrong column: same order,
        // same token, only the expiry differs.
        val live = order(tokenExpiresAt = OffsetDateTime.now().plusDays(1))
        stub(live)
        service.prepare(live.id, token)

        val dead = order(tokenExpiresAt = OffsetDateTime.now().minusSeconds(1))
        stub(dead)
        assertFailsWith<NotFoundException> { service.prepare(dead.id, token) }
    }

    @Test
    fun `a mismatched token is still refused inside the window`() {
        val order = order(tokenExpiresAt = OffsetDateTime.now().plusDays(30))
        stub(order)

        assertFailsWith<NotFoundException> { service.prepare(order.id, "b".repeat(64)) }
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private fun order(tokenExpiresAt: OffsetDateTime): Order = Order(
        eventId = UUID.randomUUID(),
        recipientEmail = "runner@test.local",
        paymentMethodWire = PaymentMethod.GCASH.wire,
        status = OrderStatus.PAID,
        totalPhp = BigDecimal("125.00"),
        shareToken = token,
        tokenExpiresAt = tokenExpiresAt,
    )

    /** One PAID order, one entitled photo, grant live for another year. */
    private fun stub(order: Order) {
        val photo = Photo(
            eventId = order.eventId,
            s3Key = "photos/${UUID.randomUUID()}.jpg",
            pricePhp = BigDecimal("125.00"),
        )
        Mockito.`when`(orderRepository.findById(order.id)).thenReturn(Optional.of(order))
        Mockito.`when`(orderItemRepository.findByIdOrderId(order.id)).thenReturn(
            listOf(
                OrderItem(
                    id = OrderItemId(orderId = order.id, photoId = photo.id),
                    pricePhpAtPurchase = BigDecimal("125.00"),
                ),
            ),
        )
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(photo))
        Mockito.`when`(downloadGrantRepository.findByIdOrderId(order.id)).thenReturn(
            listOf(
                DownloadGrant(
                    id = DownloadGrantId(orderId = order.id, photoId = photo.id),
                    grantedUntil = OffsetDateTime.now().plusYears(1),
                ),
            ),
        )
        Mockito.`when`(eventRepository.findById(order.eventId)).thenReturn(
            Optional.of(
                Event(
                    slug = "cebu-marathon",
                    name = "Cebu Marathon",
                    date = LocalDate.of(2026, 1, 11),
                    location = "Cebu City, Cebu",
                    status = EventStatus.COMPLETED,
                ),
            ),
        )
    }

    private fun <T> anyArg(): T = Mockito.any()
}
