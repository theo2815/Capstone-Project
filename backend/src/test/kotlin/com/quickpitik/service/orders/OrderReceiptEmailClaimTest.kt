package com.quickpitik.service.orders

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.config.ResendProperties
import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.email.ResendSendEmailRequest
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
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageRequest
import org.springframework.transaction.TransactionDefinition
import org.springframework.transaction.support.AbstractPlatformTransactionManager
import org.springframework.transaction.support.DefaultTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

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
    fun `a failed send leaves its claim for the stale sweep`() {
        val order = stubReadyOrder()
        stubClaimResult(order, won = true)
        Mockito.`when`(resendClient.send(anyArg())).thenThrow(RuntimeException("resend down"))

        service().sendReceiptIfPending(order.id)

        Mockito.verify(orderRepository).findById(order.id)
        Mockito.verify(orderRepository).claimReceiptSend(eqArg(order.id), anyArg())
        Mockito.verifyNoMoreInteractions(orderRepository)
    }

    @Test
    fun `a successful send stamps delivery`() {
        val order = stubReadyOrder()
        stubClaimResult(order, won = true)
        Mockito.`when`(resendClient.send(anyArg())).thenReturn(ResendSendEmailResponse(id = "re_123"))

        service().sendReceiptIfPending(order.id)

        Mockito.verify(orderRepository).stampReceiptDelivered(eqArg(order.id), anyArg())
    }

    @Test
    fun `a successful send stamps delivery in a separate transaction`() {
        val tx = TxTracker()
        val order = stubReadyOrder()
        stubClaimResult(order, won = true)
        var sendReturned = false
        Mockito.`when`(resendClient.send(anyArg())).thenAnswer {
            sendReturned = true
            ResendSendEmailResponse(id = "re_123")
        }
        var openAtStamp: Boolean? = null
        var sendHadReturnedAtStamp: Boolean? = null
        Mockito.`when`(orderRepository.stampReceiptDelivered(eqArg(order.id), eqArg(NOW))).thenAnswer {
            openAtStamp = tx.open
            sendHadReturnedAtStamp = sendReturned
            1
        }

        service(tx.template).sendReceiptIfPending(order.id)

        assertEquals(true, openAtStamp, "the delivery stamp must commit after Resend returns")
        assertEquals(true, sendHadReturnedAtStamp, "delivery cannot be stamped before Resend succeeds")
    }

    @Test
    fun `the stale sweep reads at most fifty claims older than ten minutes and retries released claims`() {
        val order = stubReadyOrder()
        Mockito.`when`(
            orderRepository.findStaleReceiptClaimIds(CUTOFF, PageRequest.of(0, 50)),
        ).thenReturn(listOf(order.id))
        Mockito.`when`(orderRepository.releaseStaleReceiptSend(order.id, CUTOFF)).thenReturn(1)
        stubClaimResult(order, won = true)
        Mockito.`when`(resendClient.send(anyArg())).thenReturn(ResendSendEmailResponse(id = "re_123"))

        service().recoverStaleReceiptClaims()

        Mockito.verify(resendClient).send(anyArg())
    }

    @Test
    fun `the stale sweep does not retry a claim another instance already reclaimed`() {
        val orderId = UUID.randomUUID()
        Mockito.`when`(
            orderRepository.findStaleReceiptClaimIds(CUTOFF, PageRequest.of(0, 50)),
        ).thenReturn(listOf(orderId))
        Mockito.`when`(orderRepository.releaseStaleReceiptSend(orderId, CUTOFF)).thenReturn(0)

        service().recoverStaleReceiptClaims()

        Mockito.verify(orderRepository, Mockito.never()).findById(orderId)
        Mockito.verifyNoInteractions(resendClient)
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

    // Free checkout (2026-09-05): a ₱0 order gets the same receipt; it just
    // says nothing was paid and shows the list price the coupon waived.
    @Test
    fun `a free order's receipt confirms the order and shows the waived price`() {
        val order = Order(
            eventId = UUID.randomUUID(),
            recipientEmail = "runner@test.local",
            paymentMethodWire = PaymentMethod.QRPH.wire,
            status = OrderStatus.FULFILLED,
            totalPhp = BigDecimal("0.00"),
            couponCode = "FREE100",
        )

        val html = service().renderHtml(
            order, "Cebu Marathon", 1, "https://api.test/bundle", BigDecimal("125.00"), BigDecimal("125.00"),
        )

        assertTrue(html.contains("QuickPitik · Order confirmed"))
        assertTrue(html.contains("List ₱125.00 · FREE100 −₱125.00 · Total ₱0.00. Nothing was charged."))
        assertTrue(html.contains("Confirmed · <span"))
        assertFalse(html.contains("Payment received"))
    }

    @Test
    fun `a paid coupon order's receipt keeps the payment wording and adds the discount line`() {
        val order = Order(
            eventId = UUID.randomUUID(),
            recipientEmail = "runner@test.local",
            paymentMethodWire = PaymentMethod.QRPH.wire,
            status = OrderStatus.FULFILLED,
            totalPhp = BigDecimal("106.25"),
        )

        val html = service().renderHtml(
            order, "Cebu Marathon", 1, "https://api.test/bundle", BigDecimal("125.00"), BigDecimal("18.75"),
        )

        assertTrue(html.contains("QuickPitik · Payment received"))
        assertTrue(html.contains("List ₱125.00 · Photographer discount −₱18.75 · Total ₱106.25."))
        assertFalse(html.contains("Nothing was charged"))
    }

    // The email's action button must open the in-app /orders/return page (which
    // runs the mobile-safe save flow), not the raw download-bundle API.
    @Test
    fun `receipt button links to the in-app receipt page, not the raw download API`() {
        val order = stubReadyOrder()
        stubClaimResult(order, won = true)
        var sent: ResendSendEmailRequest? = null
        Mockito.`when`(resendClient.send(anyArg())).thenAnswer { inv ->
            sent = inv.getArgument<ResendSendEmailRequest>(0)
            ResendSendEmailResponse(id = "re_123")
        }

        service().sendReceiptIfPending(order.id)

        val html = sent?.html ?: error("receipt was not sent")
        assertTrue(html.contains("orders/return?orderId=${order.id}"), html)
        assertTrue(html.contains("token="), html)
        assertFalse(html.contains("/download-bundle"), "email must not link the raw download API")
        assertFalse(
            Regex("""<a\b[^>]*\bdownload\b""").containsMatchIn(html),
            "the page link must not carry a download attribute",
        )
    }

    // P1.1 (2026-09-11 architecture plan): the send used to run inside a
    // REQUIRES_NEW transaction, so Resend's three attempts with Thread.sleep
    // backoff pinned one of Hikari's ten connections for the whole outage —
    // eight concurrent listeners could starve the pool at race finish. The
    // claim still has to be transactional; the HTTP call must not be.
    @Test
    fun `the receipt is sent outside the transaction that claims it`() {
        val tx = TxTracker()
        val order = stubReadyOrder()
        var openAtClaim: Boolean? = null
        var openAtSend: Boolean? = null
        Mockito.`when`(orderRepository.claimReceiptSend(eqArg(order.id), anyArg())).thenAnswer {
            openAtClaim = tx.open
            1
        }
        Mockito.`when`(resendClient.send(anyArg())).thenAnswer {
            openAtSend = tx.open
            ResendSendEmailResponse(id = "re_123")
        }

        service(tx.template).sendReceiptIfPending(order.id)

        assertEquals(true, openAtClaim, "the claim is a @Modifying UPDATE — it must stay in the transaction")
        assertEquals(false, openAtSend, "Resend must not be called while a DB connection is held")
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun service(transactionTemplate: TransactionTemplate = testTransactionTemplate()) =
        OrderReceiptEmailService(
            orderRepository,
            orderItemRepository,
            eventRepository,
            downloadGrantRepository,
            resendClient,
            ResendProperties(),
            PaymongoProperties(),
            OrderAccessTokenService(PlatformProperties()),
            transactionTemplate,
            CLOCK,
            Mockito.mock(com.quickpitik.repository.PhotoRepository::class.java),
        )

    /** A TransactionTemplate that reports whether it currently has one open. */
    private class TxTracker {
        var open = false
        val template: TransactionTemplate = TransactionTemplate(
            object : AbstractPlatformTransactionManager() {
                override fun doGetTransaction(): Any = Any()
                override fun doBegin(transaction: Any, definition: TransactionDefinition) {
                    open = true
                }

                override fun doCommit(status: DefaultTransactionStatus) {
                    open = false
                }

                override fun doRollback(status: DefaultTransactionStatus) {
                    open = false
                }
            },
        )
    }

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

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2026-09-16T12:00:00Z")
        val CUTOFF: OffsetDateTime = NOW.minusMinutes(10)
        val CLOCK: Clock = Clock.fixed(Instant.parse("2026-09-16T12:00:00Z"), ZoneOffset.UTC)
    }
}
