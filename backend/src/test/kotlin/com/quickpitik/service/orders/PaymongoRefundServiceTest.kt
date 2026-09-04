package com.quickpitik.service.orders

import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.orders.PaymongoRefundAttributes
import com.quickpitik.dto.orders.PaymongoRefundRequest
import com.quickpitik.dto.orders.PaymongoRefundResource
import com.quickpitik.dto.orders.PaymongoRefundResponse
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeResolution
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.DownloadGrantId
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Payment
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Transaction
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull

class PaymongoRefundServiceTest {
    // ₱150 list, PHOTO20 (20% of the photographer's ₱112.50 share = ₱22.50):
    // the runner paid ₱127.50, the ledger holds kept 90.00 / discount 22.50.
    @Test
    fun `a full refund of a coupon item returns what the runner paid and reverses the ledger row exactly`() {
        val h = couponHarness()

        h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_FULL, null, "Duplicate charge")

        assertEquals(12_750L, h.sentRequest!!.data.attributes.amount)
        val refund = h.minted.single()
        assertEquals(BigDecimal("-90.00"), refund.amountKeptPhp)
        assertEquals(BigDecimal("-22.50"), refund.discountPhp)
        assertEquals(h.original.id, refund.refundOf)
    }

    @Test
    fun `a partial refund of a coupon item reverses kept and discount in proportion`() {
        val h = couponHarness(providerAmount = 6_375L)

        h.service.request(h.adminId, h.dispute.id, DisputeResolution.REFUND_PARTIAL, BigDecimal("63.75"), null)

        assertEquals(6_375L, h.sentRequest!!.data.attributes.amount)
        val refund = h.minted.single()
        assertEquals(BigDecimal("-45.00"), refund.amountKeptPhp)
        assertEquals(BigDecimal("-11.25"), refund.discountPhp)
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

    private class CouponHarness(
        val service: PaymongoRefundService,
        val adminId: UUID,
        val dispute: Dispute,
        val original: Transaction,
        val client: PaymongoClient,
        val minted: List<Transaction>,
        val sentRequestRef: () -> PaymongoRefundRequest?,
    ) {
        val sentRequest: PaymongoRefundRequest? get() = sentRequestRef()
    }

    private fun couponHarness(providerAmount: Long = 12_750L): CouponHarness {
        val adminId = UUID.randomUUID()
        val photographerId = UUID.randomUUID()
        val photo = Photo(eventId = UUID.randomUUID(), s3Key = "photos/test.jpg", pricePhp = BigDecimal("150.00"))
            .also { it.photographerId = photographerId }
        val order = Order(
            eventId = photo.eventId,
            recipientEmail = "runner@example.com",
            paymentMethodWire = PaymentMethod.GCASH.wire,
            status = OrderStatus.FULFILLED,
            totalPhp = BigDecimal("127.50"),
            couponCode = "PHOTO20",
        )
        val dispute = Dispute(orderId = order.id, photoId = photo.id, reasonWire = "other")
        val payment = Payment(
            orderId = order.id,
            provider = "paymongo",
            providerRef = "cs_test",
            providerPaymentId = "pay_test",
            amountPhp = order.totalPhp,
            status = PaymentStatus.SUCCEEDED,
        )
        val original = Transaction(
            paidAt = OffsetDateTime.now(),
            photographerId = photographerId,
            eventId = order.eventId,
            photoId = photo.id,
            orderId = order.id,
            amountKeptPhp = BigDecimal("90.00"),
            discountPhp = BigDecimal("22.50"),
        )
        val disputes = Mockito.mock(DisputeRepository::class.java)
        val orders = Mockito.mock(OrderRepository::class.java)
        val items = Mockito.mock(OrderItemRepository::class.java)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val photos = Mockito.mock(PhotoRepository::class.java)
        val transactions = Mockito.mock(TransactionRepository::class.java)
        val client = Mockito.mock(PaymongoClient::class.java)
        val decisions = Mockito.mock(AdminDecisionLogService::class.java)
        Mockito.`when`(disputes.findByIdForUpdate(dispute.id)).thenReturn(dispute)
        Mockito.`when`(disputes.findByOrderId(order.id)).thenReturn(listOf(dispute))
        Mockito.`when`(items.findByIdOrderId(order.id)).thenReturn(
            listOf(OrderItem(OrderItemId(order.id, photo.id), photo.pricePhp, discountPhp = BigDecimal("22.50"))),
        )
        Mockito.`when`(payments.findByOrderId(order.id)).thenReturn(listOf(payment))
        Mockito.`when`(orders.findByIdForUpdate(order.id)).thenReturn(order)
        Mockito.`when`(orders.findById(order.id)).thenReturn(Optional.of(order))
        Mockito.`when`(photos.findById(photo.id)).thenReturn(Optional.of(photo))
        Mockito.`when`(transactions.findByOrderIdAndPhotoIdAndIsRefund(order.id, photo.id, false)).thenReturn(original)
        Mockito.`when`(transactions.findByOrderIdAndPhotoIdAndIsRefund(order.id, photo.id, true)).thenReturn(null)
        val minted = mutableListOf<Transaction>()
        Mockito.`when`(transactions.save(anyArg())).thenAnswer { call ->
            (call.arguments[0] as Transaction).also { minted += it }
        }
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
        val service = PaymongoRefundService(
            disputes,
            orders,
            items,
            payments,
            Mockito.mock(DownloadGrantRepository::class.java),
            photos,
            transactions,
            PlatformProperties(),
            client,
            decisions,
            Mockito.mock(RunnerMessagesService::class.java),
            testTransactionTemplate(),
        )
        return CouponHarness(service, adminId, dispute, original, client, minted) { sentRequest }
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
        val transactions = Mockito.mock(TransactionRepository::class.java)
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
            transactions,
            PlatformProperties(),
            client,
            decisions,
            runnerMessages,
            testTransactionTemplate(),
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
            Mockito.mock(TransactionRepository::class.java),
            PlatformProperties(),
            client,
            Mockito.mock(AdminDecisionLogService::class.java),
            Mockito.mock(RunnerMessagesService::class.java),
            testTransactionTemplate(),
        )

        service.reconcile()

        assertEquals("manual_review", dispute.refundStatus)
        assertEquals(DisputeStatus.ESCALATED, dispute.status)
        Mockito.verifyNoInteractions(client)
    }

    private fun <T> anyArg(): T = Mockito.any()
}
