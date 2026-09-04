package com.quickpitik.service.orders

import com.quickpitik.dto.orders.PaymongoEventAttributes
import com.quickpitik.dto.orders.PaymongoEventData
import com.quickpitik.dto.orders.PaymongoResource
import com.quickpitik.dto.orders.PaymongoResourceAttributes
import com.quickpitik.dto.orders.PaymongoWebhookEvent
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Payment
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.service.earnings.TransactionMintingService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import java.math.BigDecimal
import java.util.UUID
import kotlin.test.assertEquals

class PaymongoWebhookServiceTest {
    @Test
    fun `paid delivery locks and fulfills once while retaining the payment id`() {
        val order = Order(
            eventId = UUID.randomUUID(),
            recipientEmail = "runner@example.com",
            paymentMethodWire = PaymentMethod.QRPH.wire,
            totalPhp = BigDecimal("125.00"),
        )
        val photoId = UUID.randomUUID()
        val payment = Payment(
            orderId = order.id,
            provider = "paymongo",
            providerRef = "pi_test",
            amountPhp = order.totalPhp,
        )
        val orders = Mockito.mock(OrderRepository::class.java)
        val items = Mockito.mock(OrderItemRepository::class.java)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val grants = Mockito.mock(DownloadGrantRepository::class.java)
        val minting = Mockito.mock(TransactionMintingService::class.java)
        val publisher = Mockito.mock(ApplicationEventPublisher::class.java)
        val refunds = Mockito.mock(PaymongoRefundService::class.java)
        Mockito.`when`(payments.findAllByProviderAndProviderRefForUpdate("paymongo", "pi_test"))
            .thenReturn(listOf(payment))
        Mockito.`when`(orders.findByIdForUpdate(order.id)).thenReturn(order)
        Mockito.`when`(items.findByIdOrderId(order.id)).thenReturn(
            listOf(OrderItem(OrderItemId(order.id, photoId), BigDecimal("125.00"))),
        )
        Mockito.`when`(grants.findByIdOrderId(order.id)).thenReturn(emptyList())
        val service = PaymongoWebhookService(orders, items, payments, grants, minting, publisher, refunds)

        val first = service.handle(paidEvent())
        val duplicate = service.handle(paidEvent())

        assertEquals(PaymentStatus.SUCCEEDED, payment.status)
        assertEquals("pay_test", payment.providerPaymentId)
        assertEquals(OrderStatus.FULFILLED, order.status)
        assertEquals(true, first["applied"])
        assertEquals(false, duplicate["applied"])
        Mockito.verify(grants, Mockito.times(1)).save(anyArg())
        Mockito.verify(minting, Mockito.times(1)).mintForPaidOrder(order.id)
        Mockito.verify(payments, Mockito.times(2))
            .findAllByProviderAndProviderRefForUpdate("paymongo", "pi_test")
    }

    private fun paidEvent() = PaymongoWebhookEvent(
        PaymongoEventData(
            attributes = PaymongoEventAttributes(
                type = "payment.paid",
                data = PaymongoResource(
                    id = "pay_test",
                    attributes = PaymongoResourceAttributes(
                        paymentIntentId = "pi_test",
                    ),
                ),
            ),
        ),
    )

    private fun <T> anyArg(): T = Mockito.any()
}
