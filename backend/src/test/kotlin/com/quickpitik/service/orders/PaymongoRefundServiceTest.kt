package com.quickpitik.service.orders

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.dto.orders.PaymongoRefundAttributes
import com.quickpitik.dto.orders.PaymongoRefundRequest
import com.quickpitik.dto.orders.PaymongoRefundResource
import com.quickpitik.dto.orders.PaymongoRefundResponse
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponse
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponseEnvelope
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponseAttributes
import com.quickpitik.dto.orders.PaymongoPaymentResource
import com.quickpitik.dto.orders.PaymongoPaymentAttributes
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeResolution
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.DownloadGrantId
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.RunnerMessageKind
import com.quickpitik.entity.Payment
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.entity.Photo
import com.quickpitik.exception.ValidationException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.earnings.TransactionMintingService
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpStatus
import org.springframework.web.client.HttpClientErrorException
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.time.temporal.ChronoUnit
import java.nio.charset.StandardCharsets
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class PaymongoRefundServiceTest {
    // ₱150 list, PHOTO20 (20% of the photographer's ₱112.50 share = ₱22.50):
    // the runner paid ₱127.50. The ledger reversal itself is TransactionMintingService's
    // ("a full refund negates the sale row exactly"); here we only prove the
    // charged amount is what reaches both PayMongo and the ledger.
    @Test
    fun `a full refund of a coupon item returns what the runner paid and hands it to the ledger`() {
        val h = couponHarness()

        h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, "Duplicate charge")

        assertEquals(12_750L, h.sentRequest!!.data.attributes.amount)
        Mockito.verify(h.minting).mintRefund(h.dispute, BigDecimal("127.50"), h.photographerId)
    }

    @Test
    fun `a partial refund of a coupon item returns only the requested amount`() {
        val h = couponHarness(providerAmount = 6_375L)

        h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_PARTIAL, BigDecimal("63.75"), null)

        assertEquals(6_375L, h.sentRequest!!.data.attributes.amount)
        Mockito.verify(h.minting).mintRefund(h.dispute, BigDecimal("63.75"), h.photographerId)
    }

    @Test
    fun `a partial refund preserves settlement status unless no longer fully refunded`() {
        for ((before, after) in listOf(
            OrderStatus.PAID to OrderStatus.PAID,
            OrderStatus.FULFILLED to OrderStatus.FULFILLED,
            OrderStatus.REFUNDED to OrderStatus.FULFILLED,
        )) {
            val h = couponHarness(providerAmount = 6_375L)
            h.order.status = before

            h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_PARTIAL, BigDecimal("63.75"), null)

            assertEquals(DisputeStatus.RESOLVED, h.dispute.status)
            assertEquals(after, h.order.status)
        }
    }

    @Test
    fun `a partial refund cannot exceed what the runner actually paid`() {
        val h = couponHarness()

        // 130.00 is below the ₱150 list price but above the ₱127.50 charged.
        assertFailsWith<ValidationException> {
            h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_PARTIAL, BigDecimal("130.00"), null)
        }
        Mockito.verifyNoInteractions(h.client)
    }

    @Test
    fun `a QR Ph refund replaces a stale payment ID from its Payment Intent`() {
        val h = couponHarness(providerRef = "pi_test", providerPaymentId = "pay_failed")
        Mockito.`when`(h.client.retrievePaymentIntent("pi_test")).thenReturn(
            PaymongoPaymentIntentResponse(
                PaymongoPaymentIntentResponseEnvelope(
                    id = "pi_test",
                    attributes = PaymongoPaymentIntentResponseAttributes(
                        status = "succeeded",
                        payments = listOf(
                            PaymongoPaymentResource(id = "pay_failed", attributes = PaymongoPaymentAttributes(status = "failed")),
                            PaymongoPaymentResource(id = "pay_paid", attributes = PaymongoPaymentAttributes(status = "paid")),
                        ),
                    ),
                ),
            ),
        )

        h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, null)

        assertEquals("pay_paid", h.sentRequest!!.data.attributes.paymentId)
        Mockito.verify(h.client, Mockito.never()).retrieveCheckoutSession("pi_test")
    }

    @Test
    fun `a rejected refund is escalated instead of retried for hours`() {
        val h = couponHarness()
        Mockito.`when`(h.client.createRefund(anyArg(), anyArg())).thenAnswer {
            throw HttpClientErrorException.create(
                HttpStatus.BAD_REQUEST, "Bad Request", HttpHeaders(),
                """{"errors":[{"code":"parameter_invalid","detail":"Refunds are not allowed for payments with source type qrph"}]}"""
                    .toByteArray(StandardCharsets.UTF_8), StandardCharsets.UTF_8,
            )
        }

        assertFailsWith<ConflictException> {
            h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, null)
        }

        assertEquals("manual_review", h.dispute.refundStatus)
        assertEquals(DisputeStatus.ESCALATED, h.dispute.status)
    }

    private fun refusedRefundHarness(): CouponHarness {
        val h = couponHarness()
        Mockito.`when`(h.client.createRefund(anyArg(), anyArg())).thenAnswer {
            throw HttpClientErrorException.create(
                HttpStatus.BAD_REQUEST, "Bad Request", HttpHeaders(),
                """{"errors":[{"code":"parameter_invalid","detail":"Refunds are not allowed for payments with source type qrph"}]}"""
                    .toByteArray(StandardCharsets.UTF_8), StandardCharsets.UTF_8,
            )
        }
        return h
    }

    @Test
    fun `a refused refund records one hand-off row with no reason and no final resolution`() {
        val h = refusedRefundHarness()

        assertFailsWith<ConflictException> {
            h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, "checked")
        }

        Mockito.verify(h.decisions, Mockito.times(1)).logDisputeDecision(
            h.adminId,
            h.dispute.id,
            "refund_sent_to_manual_review",
            null,
            mapOf("requestedResolution" to "refund_full", "refundAmount" to "127.50"),
            null,
        )
    }

    // INC-12: Postgres keeps microseconds and may hand the time back in another offset, so the
    // reloaded row must still count as the same request. The offset always changes, so the test
    // catches an exact-equality check on any machine's time zone.
    private fun reloadFromDatabase(dispute: Dispute) {
        val stored = dispute.refundRequestedAt!!
        val otherOffset = if (stored.offset == ZoneOffset.UTC) ZoneOffset.ofHours(8) else ZoneOffset.UTC
        dispute.refundRequestedAt = stored.truncatedTo(ChronoUnit.MICROS).withOffsetSameInstant(otherOffset)
    }

    @Test
    fun `a refused refund goes to manual review at once after the database round trip`() {
        val h = couponHarness()
        Mockito.`when`(h.client.createRefund(anyArg(), anyArg())).thenAnswer {
            reloadFromDatabase(h.dispute)
            throw HttpClientErrorException.create(
                HttpStatus.BAD_REQUEST, "Bad Request", HttpHeaders(),
                """{"errors":[{"code":"parameter_invalid","detail":"Refunds are not allowed for payments with source type qrph"}]}"""
                    .toByteArray(StandardCharsets.UTF_8), StandardCharsets.UTF_8,
            )
        }

        assertFailsWith<ConflictException> {
            h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, null)
        }

        assertEquals("manual_review", h.dispute.refundStatus)
        Mockito.verify(h.decisions, Mockito.times(1)).logDisputeDecision(
            anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(),
        )
    }

    @Test
    fun `a refund PayMongo settles at once is saved after the database round trip`() {
        val h = couponHarness()
        Mockito.`when`(h.client.createRefund(anyArg(), anyArg())).thenAnswer {
            reloadFromDatabase(h.dispute)
            PaymongoRefundResponse(
                PaymongoRefundResource(
                    id = "ref_test",
                    attributes = PaymongoRefundAttributes(amount = 12_750L, paymentId = "pay_test", status = "succeeded"),
                ),
            )
        }

        h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, null)

        assertEquals("succeeded", h.dispute.refundStatus)
        assertEquals("ref_test", h.dispute.providerRefundId)
    }

    @Test
    fun `a worker that lost the race to manual review writes no second row`() {
        val h = couponHarness()
        // Another worker already moved the dispute to manual review while the 400 was in flight.
        Mockito.`when`(h.client.createRefund(anyArg(), anyArg())).thenAnswer {
            h.dispute.refundStatus = "manual_review"
            throw HttpClientErrorException.create(
                HttpStatus.BAD_REQUEST, "Bad Request", HttpHeaders(),
                """{"errors":[{"code":"parameter_invalid","detail":"Refunds are not allowed for payments with source type qrph"}]}"""
                    .toByteArray(StandardCharsets.UTF_8), StandardCharsets.UTF_8,
            )
        }

        assertFailsWith<ConflictException> {
            h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, null)
        }

        Mockito.verifyNoInteractions(h.decisions)
    }

    @Test
    fun `a refused refund with no requesting admin still goes to manual review without a row`() {
        val h = refusedRefundHarness()
        // prepare sets refundRequestedBy; simulate a legacy row by clearing it before the failure path runs
        Mockito.doAnswer {
            h.dispute.refundRequestedBy = null
            throw HttpClientErrorException.create(
                HttpStatus.BAD_REQUEST, "Bad Request", HttpHeaders(),
                """{"errors":[{"code":"parameter_invalid","detail":"Refunds are not allowed for payments with source type qrph"}]}"""
                    .toByteArray(StandardCharsets.UTF_8), StandardCharsets.UTF_8,
            )
        }.`when`(h.client).createRefund(anyArg(), anyArg())

        assertFailsWith<ConflictException> {
            h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, null)
        }

        assertEquals("manual_review", h.dispute.refundStatus)
        Mockito.verifyNoInteractions(h.decisions)
    }

    @Test
    fun `a delayed failed response cannot overwrite a completed retry`() {
        val h = couponHarness()
        h.dispute.apply {
            resolution = DisputeResolution.REFUND_FULL
            refundAmountPhp = BigDecimal("127.50")
            refundStatus = "processing"
            refundRequestedAt = OffsetDateTime.now().minusMinutes(1)
            refundRequestedBy = h.adminId
            providerRefundId = "ref_old"
        }
        Mockito.`when`(h.disputes.findByRefundStatusInOrderByRefundRequestedAtAsc(anyArg(), anyArg()))
            .thenReturn(listOf(h.dispute))
        Mockito.`when`(h.client.retrieveRefund("ref_old")).thenAnswer {
            h.dispute.refundStatus = "failed"
            h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, null)
            PaymongoRefundResponse(
                PaymongoRefundResource(
                    id = "ref_old",
                    attributes = PaymongoRefundAttributes(
                        amount = 12_750,
                        paymentId = "pay_test",
                        status = "failed",
                    ),
                ),
            )
        }

        h.service.reconcile()

        assertEquals("ref_test", h.dispute.providerRefundId)
        assertEquals("succeeded", h.dispute.refundStatus)
        assertEquals(DisputeStatus.RESOLVED, h.dispute.status)
        Mockito.verify(h.client, Mockito.times(1)).createRefund(anyArg(), anyArg())
    }

    @Test
    fun `a stale webhook cannot downgrade a completed refund`() {
        val h = couponHarness()
        h.dispute.apply {
            resolution = DisputeResolution.REFUND_FULL
            refundAmountPhp = BigDecimal("127.50")
            refundStatus = "succeeded"
            refundRequestedAt = OffsetDateTime.now()
            refundedAt = OffsetDateTime.now()
            providerRefundId = "ref_current"
            status = DisputeStatus.RESOLVED
        }
        Mockito.`when`(h.disputes.findByProviderRefundId("ref_old")).thenReturn(h.dispute)

        val handled = h.service.handleWebhook("ref_old", "failed")

        assertEquals(false, handled)
        assertEquals("ref_current", h.dispute.providerRefundId)
        assertEquals("succeeded", h.dispute.refundStatus)
    }

    private val qrphRefusal = "PayMongo's dashboard can only refund the whole payment. Deny this request with a reason instead."

    @Test
    fun `approving a whole QR Ph payment parks it for a manual refund and never calls PayMongo`() {
        val h = couponHarness(method = PaymentMethod.QRPH, providerRef = "pi_test")
        val decision = AdminDecisionLog(adminId = h.adminId, targetDisputeId = h.dispute.id, decision = "refund_sent_to_manual_review")
        Mockito.`when`(h.decisions.logDisputeDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(decision)

        h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, "  QR Ph  ")

        Mockito.verifyNoInteractions(h.client)
        assertEquals("manual_review", h.dispute.refundStatus)
        assertEquals(DisputeStatus.ESCALATED, h.dispute.status)
        assertEquals(h.adminId, h.dispute.refundRequestedBy)
        assertEquals(BigDecimal("127.50"), h.dispute.refundAmountPhp)
        assertEquals("QR Ph", h.dispute.refundReason)
        Mockito.verify(h.decisions, Mockito.times(1)).logDisputeDecision(
            h.adminId,
            h.dispute.id,
            "refund_sent_to_manual_review",
            null,
            mapOf("requestedResolution" to "refund_full", "refundAmount" to "127.50"),
            null,
        )
        Mockito.verifyNoMoreInteractions(h.decisions)
        Mockito.verify(h.runnerMessages).pushMessage(
            h.runnerId,
            RunnerMessageKind.ADMIN_MESSAGE,
            "Your refund of PHP 127.50 was approved. We will send you a PayMongo link to claim it.",
            h.adminId,
            decision.id,
            h.order.id,
            "Refund approved",
        )
        Mockito.verifyNoMoreInteractions(h.runnerMessages)
        Mockito.verifyNoInteractions(h.minting)
        Mockito.verifyNoInteractions(h.grants)
    }

    @Test
    fun `a partial refund of a QR Ph payment is refused because the dashboard cannot do partial refunds`() {
        val h = couponHarness(method = PaymentMethod.QRPH, providerRef = "pi_test")

        val error = assertFailsWith<ConflictException> {
            h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_PARTIAL, BigDecimal("63.75"), null)
        }

        assertEquals(qrphRefusal, error.message)
        assertRefusedWithoutChanges(h)
    }

    @Test
    fun `a full refund of one photo is refused when the QR Ph checkout paid for more`() {
        val h = couponHarness(method = PaymentMethod.QRPH, providerRef = "pi_test", siblingPaymentPhp = BigDecimal("100.00"))

        val error = assertFailsWith<ConflictException> {
            h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, null)
        }

        assertEquals(qrphRefusal, error.message)
        assertRefusedWithoutChanges(h)
    }

    private fun assertRefusedWithoutChanges(h: CouponHarness) {
        Mockito.verify(h.disputes, Mockito.never()).save(h.dispute)
        assertEquals(null, h.dispute.refundStatus)
        assertEquals(DisputeStatus.OPEN, h.dispute.status)
        Mockito.verifyNoInteractions(h.client)
        Mockito.verifyNoInteractions(h.decisions)
        Mockito.verifyNoInteractions(h.runnerMessages)
    }

    @Test
    fun `a payment that is not QR Ph still asks PayMongo to refund`() {
        val h = couponHarness(method = PaymentMethod.GCASH)

        h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, null)

        Mockito.verify(h.client, Mockito.times(1)).createRefund(anyArg(), anyArg())
        assertEquals("succeeded", h.dispute.refundStatus)
    }

    private class CouponHarness(
        val service: PaymongoRefundService,
        val adminId: UUID,
        val dispute: Dispute,
        val order: Order,
        val photographerId: UUID,
        val client: PaymongoClient,
        val minting: TransactionMintingService,
        val disputes: DisputeRepository,
        val decisions: AdminDecisionLogService,
        val runnerMessages: RunnerMessagesService,
        val grants: DownloadGrantRepository,
        val runnerId: UUID,
        val events: ApplicationEventPublisher,
        val sentRequestRef: () -> PaymongoRefundRequest?,
    ) {
        val sentRequest: PaymongoRefundRequest? get() = sentRequestRef()
    }

    private fun couponHarness(
        providerAmount: Long = 12_750L,
        providerRef: String = "cs_test",
        providerPaymentId: String? = "pay_test",
        method: PaymentMethod = PaymentMethod.GCASH,
        siblingPaymentPhp: BigDecimal? = null,
        prefixes: List<String> = listOf(linkPrefix),
    ): CouponHarness {
        val adminId = UUID.randomUUID()
        val photographerId = UUID.randomUUID()
        val photo = Photo(eventId = UUID.randomUUID(), s3Key = "photos/test.jpg", pricePhp = BigDecimal("150.00"))
            .also { it.photographerId = photographerId }
        val order = Order(
            eventId = photo.eventId,
            recipientEmail = "runner@example.com",
            paymentMethodWire = method.wire,
            status = OrderStatus.FULFILLED,
            totalPhp = BigDecimal("127.50"),
            couponCode = "PHOTO20",
        )
        val runnerId = UUID.randomUUID()
        val dispute = Dispute(orderId = order.id, photoId = photo.id, runnerId = runnerId, reasonWire = "other")
        val payment = Payment(
            orderId = order.id,
            provider = "paymongo",
            providerRef = providerRef,
            providerPaymentId = providerPaymentId,
            amountPhp = order.totalPhp,
            status = PaymentStatus.SUCCEEDED,
        )
        val disputes = Mockito.mock(DisputeRepository::class.java)
        val orders = Mockito.mock(OrderRepository::class.java)
        val items = Mockito.mock(OrderItemRepository::class.java)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val photos = Mockito.mock(PhotoRepository::class.java)
        val minting = Mockito.mock(TransactionMintingService::class.java)
        val client = Mockito.mock(PaymongoClient::class.java)
        val decisions = Mockito.mock(AdminDecisionLogService::class.java)
        Mockito.`when`(disputes.findByIdForUpdate(dispute.id)).thenReturn(dispute)
        Mockito.`when`(disputes.findByOrderId(order.id)).thenReturn(listOf(dispute))
        Mockito.`when`(items.findByIdOrderId(order.id)).thenReturn(
            listOf(OrderItem(OrderItemId(order.id, photo.id), photo.pricePhp, discountPhp = BigDecimal("22.50"))),
        )
        Mockito.`when`(payments.findByOrderId(order.id)).thenReturn(listOf(payment))
        // One QR Ph payment covers the whole checkout: a sibling order's row shares the same pi_ reference.
        val sibling = siblingPaymentPhp?.let {
            Payment(orderId = UUID.randomUUID(), provider = "paymongo", providerRef = providerRef, amountPhp = it, status = PaymentStatus.SUCCEEDED)
        }
        Mockito.`when`(payments.findAllByProviderAndProviderRef("paymongo", providerRef))
            .thenReturn(listOfNotNull(payment, sibling))
        Mockito.`when`(orders.findById(order.id)).thenReturn(Optional.of(order))
        Mockito.`when`(orders.findByIdForUpdate(order.id)).thenReturn(order)
        Mockito.`when`(photos.findById(photo.id)).thenReturn(Optional.of(photo))
        var sentRequest: PaymongoRefundRequest? = null
        Mockito.`when`(client.createRefund(anyArg(), anyArg())).thenAnswer { call ->
            sentRequest = call.getArgument(0)
            PaymongoRefundResponse(
                PaymongoRefundResource(
                    id = "ref_test",
                    attributes = PaymongoRefundAttributes(
                        amount = providerAmount,
                        paymentId = "pay_test",
                        status = "succeeded",
                    ),
                ),
            )
        }
        Mockito.`when`(
            decisions.logDisputeDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()),
        ).thenReturn(AdminDecisionLog(adminId = adminId, targetDisputeId = dispute.id, decision = "resolved"))
        val grants = Mockito.mock(DownloadGrantRepository::class.java)
        val runnerMessages = Mockito.mock(RunnerMessagesService::class.java)
        val events = Mockito.mock(ApplicationEventPublisher::class.java)
        val service = PaymongoRefundService(
            disputes,
            orders,
            items,
            payments,
            grants,
            photos,
            minting,
            client,
            decisions,
            runnerMessages,
            testTransactionTemplate(),
            PaymongoProperties(refundLinkAllowedPrefixes = prefixes),
            events,
        )
        return CouponHarness(
            service, adminId, dispute, order, photographerId, client, minting, disputes, decisions,
            runnerMessages, grants, runnerId, events,
        ) { sentRequest }
    }
    @Test
    fun `successful full refund uses PayMongo then revokes the grant`() {
        val adminId = UUID.randomUUID()
        val photo = Photo(
            eventId = UUID.randomUUID(),
            s3Key = "photos/test.jpg",
            pricePhp = BigDecimal("125.00"),
        )
        val order = Order(
            eventId = photo.eventId,
            recipientEmail = "runner@example.com",
            paymentMethodWire = PaymentMethod.GCASH.wire,
            status = OrderStatus.FULFILLED,
            totalPhp = photo.pricePhp,
        )
        val dispute = Dispute(
            orderId = order.id,
            photoId = photo.id,
            reasonWire = "other",
        )
        val payment = Payment(
            orderId = order.id,
            provider = "paymongo",
            providerRef = "cs_test",
            providerPaymentId = "pay_test",
            amountPhp = order.totalPhp,
            status = PaymentStatus.SUCCEEDED,
        )
        val disputes = Mockito.mock(DisputeRepository::class.java)
        val orders = Mockito.mock(OrderRepository::class.java)
        val items = Mockito.mock(OrderItemRepository::class.java)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val grants = Mockito.mock(DownloadGrantRepository::class.java)
        val photos = Mockito.mock(PhotoRepository::class.java)
        val minting = Mockito.mock(TransactionMintingService::class.java)
        val client = Mockito.mock(PaymongoClient::class.java)
        val decisions = Mockito.mock(AdminDecisionLogService::class.java)
        val runnerMessages = Mockito.mock(RunnerMessagesService::class.java)
        Mockito.`when`(disputes.findByIdForUpdate(dispute.id)).thenReturn(dispute)
        Mockito.`when`(disputes.findByOrderId(order.id)).thenReturn(listOf(dispute))
        Mockito.`when`(items.findByIdOrderId(order.id)).thenReturn(
            listOf(OrderItem(OrderItemId(order.id, photo.id), photo.pricePhp)),
        )
        Mockito.`when`(payments.findByOrderId(order.id)).thenReturn(listOf(payment))
        Mockito.`when`(orders.findByIdForUpdate(order.id)).thenReturn(order)
        Mockito.`when`(photos.findById(photo.id)).thenReturn(Optional.of(photo))
        var sentRequest: PaymongoRefundRequest? = null
        Mockito.`when`(client.createRefund(anyArg(), anyArg())).thenAnswer { call ->
            sentRequest = call.getArgument(0)
            PaymongoRefundResponse(
                PaymongoRefundResource(
                    id = "ref_test",
                    attributes = PaymongoRefundAttributes(
                        amount = 12_500,
                        paymentId = "pay_test",
                        status = "succeeded",
                    ),
                ),
            )
        }
        Mockito.`when`(
            decisions.logDisputeDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()),
        )
            .thenReturn(AdminDecisionLog(adminId = adminId, targetDisputeId = dispute.id, decision = "resolved"))
        val service = PaymongoRefundService(
            disputes,
            orders,
            items,
            payments,
            grants,
            photos,
            minting,
            client,
            decisions,
            runnerMessages,
            testTransactionTemplate(),
            PaymongoProperties(),
            Mockito.mock(ApplicationEventPublisher::class.java),
        )

        service.request(adminId, dispute.id, DisputeResolution.REFUND_FULL, null, "Duplicate charge")

        assertEquals(DisputeStatus.RESOLVED, dispute.status)
        assertEquals("succeeded", dispute.refundStatus)
        assertEquals("ref_test", dispute.providerRefundId)
        assertNotNull(dispute.refundedAt)
        assertEquals(OrderStatus.REFUNDED, order.status)
        Mockito.verify(grants).deleteById(DownloadGrantId(order.id, photo.id))
        assertEquals(12_500L, sentRequest!!.data.attributes.amount)
        assertEquals("pay_test", sentRequest!!.data.attributes.paymentId)
    }

    @Test
    fun `stale unknown refund outcome is escalated instead of retried`() {
        val dispute = Dispute(
            orderId = UUID.randomUUID(),
            photoId = UUID.randomUUID(),
            reasonWire = "other",
        ).apply {
            resolution = DisputeResolution.REFUND_FULL
            refundAmountPhp = BigDecimal("125.00")
            refundStatus = "requesting"
            refundRequestedAt = OffsetDateTime.now().minusHours(24)
        }
        val payment = Payment(
            orderId = dispute.orderId,
            provider = "paymongo",
            providerRef = "cs_unknown",
            amountPhp = BigDecimal("125.00"),
            status = PaymentStatus.SUCCEEDED,
        )
        val disputes = Mockito.mock(DisputeRepository::class.java)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val client = Mockito.mock(PaymongoClient::class.java)
        Mockito.`when`(disputes.findByRefundStatusInOrderByRefundRequestedAtAsc(anyArg(), anyArg()))
            .thenReturn(listOf(dispute))
        Mockito.`when`(disputes.findByIdForUpdate(dispute.id)).thenReturn(dispute)
        Mockito.`when`(payments.findByOrderId(dispute.orderId)).thenReturn(listOf(payment))
        val service = PaymongoRefundService(
            disputes,
            Mockito.mock(OrderRepository::class.java),
            Mockito.mock(OrderItemRepository::class.java),
            payments,
            Mockito.mock(DownloadGrantRepository::class.java),
            Mockito.mock(PhotoRepository::class.java),
            Mockito.mock(TransactionMintingService::class.java),
            client,
            Mockito.mock(AdminDecisionLogService::class.java),
            Mockito.mock(RunnerMessagesService::class.java),
            testTransactionTemplate(),
            PaymongoProperties(),
            Mockito.mock(ApplicationEventPublisher::class.java),
        )

        service.reconcile()

        assertEquals("manual_review", dispute.refundStatus)
        assertEquals(DisputeStatus.ESCALATED, dispute.status)
        Mockito.verifyNoInteractions(client)
    }

    private val linkPrefix = "https://transfer.test/"
    private val sentLink = "https://transfer.test/abc123"

    private fun parkedHarness(prefixes: List<String> = listOf(linkPrefix)): CouponHarness {
        val h = couponHarness(method = PaymentMethod.QRPH, providerRef = "pi_test", prefixes = prefixes)
        h.dispute.refundStatus = "manual_review"
        h.dispute.status = DisputeStatus.ESCALATED
        h.dispute.refundAmountPhp = BigDecimal("127.50")
        return h
    }

    private fun assertNothingSavedOrSent(h: CouponHarness) {
        assertNull(h.dispute.refundLinkUrl)
        assertNull(h.dispute.refundLinkSentAt)
        Mockito.verify(h.disputes, Mockito.never()).save(Mockito.any(Dispute::class.java))
        Mockito.verifyNoInteractions(h.decisions)
        Mockito.verifyNoInteractions(h.runnerMessages)
        Mockito.verifyNoInteractions(h.events)
    }

    @Test
    fun `sending a refund link saves it, logs one row, tells the runner and publishes one event`() {
        val h = parkedHarness()
        val decision = AdminDecisionLog(adminId = h.adminId, targetDisputeId = h.dispute.id, decision = "refund_link_sent")
        Mockito.`when`(h.decisions.logDisputeDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(decision)

        h.service.sendRefundLink(h.adminId, h.dispute.id, "  $sentLink ")

        assertEquals(sentLink, h.dispute.refundLinkUrl)
        assertNotNull(h.dispute.refundLinkSentAt)
        Mockito.verify(h.decisions, Mockito.times(1)).logDisputeDecision(
            h.adminId, h.dispute.id, "refund_link_sent", null, mapOf("refundLinkUrl" to sentLink), null,
        )
        Mockito.verifyNoMoreInteractions(h.decisions)
        Mockito.verify(h.runnerMessages).pushMessage(
            h.runnerId,
            RunnerMessageKind.ADMIN_MESSAGE,
            "Claim your refund of PHP 127.50 with this PayMongo link: $sentLink The link works for 3 days.",
            h.adminId,
            decision.id,
            h.order.id,
            "Your refund link is ready",
        )
        Mockito.verifyNoMoreInteractions(h.runnerMessages)
        Mockito.verify(h.events).publishEvent(RefundLinkSentEvent(h.order.id, sentLink, BigDecimal("127.50")))
        Mockito.verifyNoMoreInteractions(h.events)
        Mockito.verifyNoInteractions(h.client)
    }

    @Test
    fun `sending a second link replaces the first and sends everything again`() {
        val h = parkedHarness()
        Mockito.`when`(h.decisions.logDisputeDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = h.adminId, targetDisputeId = h.dispute.id, decision = "refund_link_sent"))
        val second = "https://transfer.test/def456"

        h.service.sendRefundLink(h.adminId, h.dispute.id, sentLink)
        val firstSentAt = assertNotNull(h.dispute.refundLinkSentAt)
        h.service.sendRefundLink(h.adminId, h.dispute.id, second)

        assertEquals(second, h.dispute.refundLinkUrl)
        assertTrue(!assertNotNull(h.dispute.refundLinkSentAt).isBefore(firstSentAt))
        Mockito.verify(h.decisions, Mockito.times(2))
            .logDisputeDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
        Mockito.verify(h.runnerMessages, Mockito.times(2))
            .pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
        Mockito.verify(h.events).publishEvent(RefundLinkSentEvent(h.order.id, sentLink, BigDecimal("127.50")))
        Mockito.verify(h.events).publishEvent(RefundLinkSentEvent(h.order.id, second, BigDecimal("127.50")))
    }

    @Test
    fun `a link of exactly 500 characters is accepted`() {
        val h = parkedHarness()
        Mockito.`when`(h.decisions.logDisputeDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = h.adminId, targetDisputeId = h.dispute.id, decision = "refund_link_sent"))
        val url = linkPrefix + "a".repeat(500 - linkPrefix.length)

        h.service.sendRefundLink(h.adminId, h.dispute.id, url)

        assertEquals(url, h.dispute.refundLinkUrl)
    }

    @Test
    fun `a dispute with no runner still saves and mails the link but gets no inbox message`() {
        val h = parkedHarness()
        h.dispute.runnerId = null
        Mockito.`when`(h.decisions.logDisputeDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = h.adminId, targetDisputeId = h.dispute.id, decision = "refund_link_sent"))

        h.service.sendRefundLink(h.adminId, h.dispute.id, sentLink)

        assertEquals(sentLink, h.dispute.refundLinkUrl)
        Mockito.verifyNoInteractions(h.runnerMessages)
        Mockito.verify(h.events).publishEvent(RefundLinkSentEvent(h.order.id, sentLink, BigDecimal("127.50")))
    }

    @Test
    fun `a refund link is refused unless the dispute waits for a manual refund`() {
        for (state in listOf(null, "requesting", "succeeded")) {
            val h = parkedHarness()
            h.dispute.refundStatus = state

            assertFailsWith<ConflictException> { h.service.sendRefundLink(h.adminId, h.dispute.id, sentLink) }

            assertNothingSavedOrSent(h)
        }
    }

    @Test
    fun `a configured prefix without a trailing slash is ignored so look-alike hosts cannot pass`() {
        val h = parkedHarness(prefixes = listOf("https://transfer.test"))

        for (url in listOf("https://transfer.test.evil.example/x", "https://transfer.test/abc")) {
            assertFailsWith<ValidationException>("link: $url") { h.service.sendRefundLink(h.adminId, h.dispute.id, url) }
        }

        assertNothingSavedOrSent(h)
    }

    @Test
    fun `a refund link for a missing dispute is a 404`() {
        val h = parkedHarness()

        assertFailsWith<NotFoundException> { h.service.sendRefundLink(h.adminId, UUID.randomUUID(), sentLink) }

        assertNothingSavedOrSent(h)
    }

    @Test
    fun `a link that is not from an allowed prefix is refused before the dispute is locked and nothing is saved or sent`() {
        val bad = listOf(
            "",
            "   ",
            "https://evil.example/x",
            "https://transfer.test.evil.example/x",
            "http://transfer.test/abc",
            "https://transfer.test/a b",
            "https://transfer.test/a\tb",
            "https://transfer.test/a\nb",
            linkPrefix + "a".repeat(500),
            // starts with the prefix and has no whitespace, but would break out of the e-mail's href
            linkPrefix + "x\"><a/href=\"https://evil.example/\">Claim</a><!--",
            linkPrefix + "x<b>",
            linkPrefix + "x\\y",
            linkPrefix + "x\u0001y",
        )
        for (url in bad) {
            val h = parkedHarness()

            val error = assertFailsWith<ValidationException>("link: $url") {
                h.service.sendRefundLink(h.adminId, h.dispute.id, url)
            }

            assertEquals("url", error.field)
            Mockito.verify(h.disputes, Mockito.never()).findByIdForUpdate(h.dispute.id)
            assertNothingSavedOrSent(h)
        }
    }

    private fun <T> anyArg(): T = Mockito.any()
}
