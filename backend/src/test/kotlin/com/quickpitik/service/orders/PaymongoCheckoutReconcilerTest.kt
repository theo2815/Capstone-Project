package com.quickpitik.service.orders

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponse
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseAttributes
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseEnvelope
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Payment
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.math.BigDecimal
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class PaymongoCheckoutReconcilerTest {
    @Test
    fun `stale active checkout is expired outside the database transaction`() {
        val order = Order(
            eventId = UUID.randomUUID(),
            recipientEmail = "runner@example.com",
            paymentMethodWire = PaymentMethod.GCASH.wire,
            totalPhp = BigDecimal("125.00"),
        )
        val payment = Payment(
            orderId = order.id,
            provider = "paymongo",
            providerRef = "cs_test",
            amountPhp = order.totalPhp,
        )
        val client = Mockito.mock(PaymongoClient::class.java)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val orders = Mockito.mock(OrderRepository::class.java)
        val webhooks = Mockito.mock(PaymongoWebhookService::class.java)
        Mockito.`when`(
            payments.findByProviderAndStatusAndProviderRefIsNotNullAndCreatedAtBeforeOrderByCreatedAtAsc(
                eqArg("paymongo"),
                eqArg(PaymentStatus.PENDING),
                anyArg(),
                anyArg(),
            ),
        ).thenReturn(listOf(payment))
        Mockito.`when`(payments.findAllByOrderIdInForUpdate(anyArg())).thenReturn(listOf(payment))
        Mockito.`when`(orders.findByIdForUpdate(order.id)).thenReturn(order)
        var providerCallWasTransactional = true
        Mockito.`when`(client.retrieveCheckoutSession("cs_test")).thenAnswer {
            providerCallWasTransactional = TransactionSynchronizationManager.isActualTransactionActive()
            checkout("active")
        }
        Mockito.`when`(client.expireCheckoutSession("cs_test")).thenReturn(checkout("expired"))
        val reconciler = PaymongoCheckoutReconciler(
            PaymongoProperties(),
            client,
            webhooks,
            payments,
            orders,
            testTransactionTemplate(),
        )

        reconciler.reconcile()

        assertFalse(providerCallWasTransactional)
        assertEquals(PaymentStatus.FAILED, payment.status)
        assertEquals(OrderStatus.EXPIRED, order.status)
        Mockito.verify(client).expireCheckoutSession("cs_test")
        Mockito.verifyNoInteractions(webhooks)
    }

    private fun checkout(status: String) = PaymongoCheckoutSessionResponse(
        PaymongoCheckoutSessionResponseEnvelope(
            id = "cs_test",
            attributes = PaymongoCheckoutSessionResponseAttributes(status = status),
        ),
    )

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
