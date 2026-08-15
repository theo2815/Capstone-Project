package com.quickpitik.service.orders

import com.quickpitik.config.PublicProperties
import com.quickpitik.config.ResendProperties
import com.quickpitik.dto.email.ResendSendEmailResponse
import com.quickpitik.entity.DownloadGrant
import com.quickpitik.entity.DownloadGrantId
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.service.email.ResendClient
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID

// Runner-audit "Checkout + webhook + email" #p1. The receipt send used a
// read-check-write on orders.email_sent_at: load the order, see null, send,
// then stamp. Two concurrent PayMongo webhook retries could both pass the
// check and both send, so the buyer was charged once and emailed twice.
//
// The claim is now a conditional UPDATE. What matters is that exactly one
// caller wins, and — critically — that a failed send hands the claim back,
// because the previous design's one redeeming property was that a transient
// Resend outage left the receipt retryable.
class OrderReceiptEmailClaimTest {

    private lateinit var orderRepository: OrderRepository
    private lateinit var orderItemRepository: OrderItemRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var downloadGrantRepository: DownloadGrantRepository
    private lateinit var resendClient: ResendClient

    @BeforeEach
    fun setUp() {
        orderRepository = Mockito.mock(OrderRepository::class.java)
        orderItemRepository = Mockito.mock(OrderItemRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        downloadGrantRepository = Mockito.mock(DownloadGrantRepository::class.java)
        resendClient = Mockito.mock(ResendClient::class.java)
    }

    @Test
    fun `winning the claim sends the receipt`() {
        val order = stubReadyOrder()
        stubClaimResult(order, won = true)
        Mockito.`when`(resendClient.send(anyArg())).thenReturn(ResendSendEmailResponse(id = "re_123"))

        service().sendReceiptIfPending(order.id)

        Mockito.verify(resendClient).send(anyArg())
    }

    @Test
    fun `losing the claim to a concurrent retry sends nothing`() {
        // The other caller's UPDATE already stamped email_sent_at, so ours
        // matches zero rows. This is the double-send that used to slip through.
        val order = stubReadyOrder()
        stubClaimResult(order, won = false)

        service().sendReceiptIfPending(order.id)

        Mockito.verify(resendClient, Mockito.never()).send(anyArg())
    }

    @Test
    fun `a failed send releases the claim so a retry can still deliver`() {
        val order = stubReadyOrder()
        stubClaimResult(order, won = true)
        Mockito.`when`(resendClient.send(anyArg())).thenThrow(RuntimeException("resend down"))

        service().sendReceiptIfPending(order.id)

        // Without this the atomic claim would turn every transient outage into
        // a permanently lost receipt.
        Mockito.verify(orderRepository).releaseReceiptSend(order.id)
    }

    @Test
    fun `a successful send keeps the claim`() {
        val order = stubReadyOrder()
        stubClaimResult(order, won = true)
        Mockito.`when`(resendClient.send(anyArg())).thenReturn(ResendSendEmailResponse(id = "re_123"))

        service().sendReceiptIfPending(order.id)

        Mockito.verify(orderRepository, Mockito.never()).releaseReceiptSend(anyArg())
    }

    @Test
    fun `an order with no download grants never burns its claim`() {
        // Grants are minted by the same webhook; if we ran early the order is
        // not ready. Claiming here would mark it sent forever.
        val order = stubReadyOrder()
        Mockito.`when`(downloadGrantRepository.findByIdOrderId(order.id)).thenReturn(emptyList())

        service().sendReceiptIfPending(order.id)

        Mockito.verify(orderRepository, Mockito.never()).claimReceiptSend(anyArg(), anyArg())
        Mockito.verify(resendClient, Mockito.never()).send(anyArg())
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun service() = OrderReceiptEmailService(
        orderRepository,
        orderItemRepository,
        eventRepository,
        downloadGrantRepository,
        resendClient,
        ResendProperties(),
        PublicProperties(),
    )

    private fun stubClaimResult(order: Order, won: Boolean) {
        Mockito.`when`(orderRepository.claimReceiptSend(eqArg(order.id), anyArg()))
            .thenReturn(if (won) 1 else 0)
    }

    /** A PAID order that clears every precondition ahead of the claim. */
    private fun stubReadyOrder(): Order {
        val order = Order(
            eventId = UUID.randomUUID(),
            recipientEmail = "runner@test.local",
            paymentMethodWire = PaymentMethod.GCASH.wire,
            status = OrderStatus.PAID,
            totalPhp = BigDecimal("125.00"),
            shareToken = "tok_" + UUID.randomUUID(),
            tokenExpiresAt = OffsetDateTime.now().plusDays(90),
        )
        val photoId = UUID.randomUUID()
        Mockito.`when`(orderRepository.findById(order.id)).thenReturn(Optional.of(order))
        Mockito.`when`(orderItemRepository.findByIdOrderId(order.id)).thenReturn(
            listOf(
                OrderItem(
                    id = OrderItemId(orderId = order.id, photoId = photoId),
                    pricePhpAtPurchase = BigDecimal("125.00"),
                ),
            ),
        )
        Mockito.`when`(downloadGrantRepository.findByIdOrderId(order.id)).thenReturn(
            listOf(
                DownloadGrant(
                    id = DownloadGrantId(orderId = order.id, photoId = photoId),
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
        return order
    }

    // Mockito.eq returns a platform type, so calling it directly in an argument
    // position of a Kotlin non-null parameter trips a null check. Same shape as
    // PhotoSearchServiceTest:188.
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value

    private fun <T> anyArg(): T = Mockito.any()
}
