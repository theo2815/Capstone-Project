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
import kotlin.test.assertNotNull

class PaymongoRefundServiceTest {
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
