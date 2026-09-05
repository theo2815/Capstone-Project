package com.quickpitik.service.orders

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponse
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseAttributes
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseEnvelope
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponse
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponseAttributes
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponseEnvelope
import com.quickpitik.dto.orders.PaymongoPaymentResource
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
            payments.findByProviderAndStatusAndProviderRefStartingWithOrderByCreatedAtAsc(
                eqArg("paymongo"),
                eqArg(PaymentStatus.PENDING),
                eqArg("pi_"),
                anyArg(),
            ),
        ).thenReturn(emptyList())
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

    @Test
    fun `reconcileOrder hands a succeeded intent to settlement outside the database transaction`() {
        val order = qrphOrder()
        val payment = Payment(orderId = order.id, provider = "paymongo", providerRef = "pi_test", amountPhp = order.totalPhp)
        val client = Mockito.mock(PaymongoClient::class.java)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val orders = Mockito.mock(OrderRepository::class.java)
        val webhooks = Mockito.mock(PaymongoWebhookService::class.java)
        Mockito.`when`(payments.findByOrderId(order.id)).thenReturn(listOf(payment))
        Mockito.`when`(payments.findAllByProviderAndProviderRef("paymongo", "pi_test")).thenReturn(listOf(payment))
        var providerCallWasTransactional = true
        Mockito.`when`(client.retrievePaymentIntent("pi_test")).thenAnswer {
            providerCallWasTransactional = TransactionSynchronizationManager.isActualTransactionActive()
            intent("succeeded")
        }
        val reconciler = PaymongoCheckoutReconciler(PaymongoProperties(), client, webhooks, payments, orders, testTransactionTemplate())

        reconciler.reconcileOrder(order.id)

        assertFalse(providerCallWasTransactional)
        Mockito.verify(webhooks).settlePaymentIntent(eqArg("pi_test"), eqArg("pay_test"), anyArg())
    }

    @Test
    fun `reconcileOrder expires a past-expiry intent`() {
        val order = qrphOrder()
        val payment = Payment(
            orderId = order.id,
            provider = "paymongo",
            providerRef = "pi_test",
            amountPhp = order.totalPhp,
            expiresAt = java.time.OffsetDateTime.now().minusMinutes(1),
        )
        val client = Mockito.mock(PaymongoClient::class.java)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val orders = Mockito.mock(OrderRepository::class.java)
        val webhooks = Mockito.mock(PaymongoWebhookService::class.java)
        Mockito.`when`(payments.findByOrderId(order.id)).thenReturn(listOf(payment))
        Mockito.`when`(payments.findAllByProviderAndProviderRef("paymongo", "pi_test")).thenReturn(listOf(payment))
        Mockito.`when`(payments.findAllByOrderIdInForUpdate(anyArg())).thenReturn(listOf(payment))
        Mockito.`when`(orders.findByIdForUpdate(order.id)).thenReturn(order)
        Mockito.`when`(client.retrievePaymentIntent("pi_test")).thenReturn(intent("awaiting_next_action"))
        val reconciler = PaymongoCheckoutReconciler(PaymongoProperties(), client, webhooks, payments, orders, testTransactionTemplate())

        reconciler.reconcileOrder(order.id)

        assertEquals(PaymentStatus.FAILED, payment.status)
        assertEquals(OrderStatus.EXPIRED, order.status)
        Mockito.verifyNoInteractions(webhooks)
    }

    @Test
    fun `reconcileOrder expires an intent whose last attachment failed`() {
        val order = qrphOrder()
        val payment = Payment(
            orderId = order.id,
            provider = "paymongo",
            providerRef = "pi_test",
            amountPhp = order.totalPhp,
            expiresAt = java.time.OffsetDateTime.now().plusMinutes(20),
        )
        val client = Mockito.mock(PaymongoClient::class.java)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val orders = Mockito.mock(OrderRepository::class.java)
        val webhooks = Mockito.mock(PaymongoWebhookService::class.java)
        Mockito.`when`(payments.findByOrderId(order.id)).thenReturn(listOf(payment))
        Mockito.`when`(payments.findAllByProviderAndProviderRef("paymongo", "pi_test")).thenReturn(listOf(payment))
        Mockito.`when`(payments.findAllByOrderIdInForUpdate(anyArg())).thenReturn(listOf(payment))
        Mockito.`when`(orders.findByIdForUpdate(order.id)).thenReturn(order)
        Mockito.`when`(client.retrievePaymentIntent("pi_test")).thenReturn(
            PaymongoPaymentIntentResponse(
                PaymongoPaymentIntentResponseEnvelope(
                    id = "pi_test",
                    attributes = PaymongoPaymentIntentResponseAttributes(
                        status = "awaiting_payment_method",
                        lastPaymentError = mapOf("code" to "generic_decline"),
                    ),
                ),
            ),
        )
        val reconciler = PaymongoCheckoutReconciler(PaymongoProperties(), client, webhooks, payments, orders, testTransactionTemplate())

        reconciler.reconcileOrder(order.id)

        assertEquals(PaymentStatus.FAILED, payment.status)
        assertEquals(OrderStatus.EXPIRED, order.status)
        Mockito.verifyNoInteractions(webhooks)
    }

    @Test
    fun `expireOrder marks one order the way a timeout would`() {
        val order = qrphOrder()
        val payment = Payment(orderId = order.id, provider = "paymongo", providerRef = "pi_test", amountPhp = order.totalPhp)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val orders = Mockito.mock(OrderRepository::class.java)
        val client = Mockito.mock(PaymongoClient::class.java)
        Mockito.`when`(payments.findAllByOrderIdInForUpdate(anyArg())).thenReturn(listOf(payment))
        Mockito.`when`(orders.findByIdForUpdate(order.id)).thenReturn(order)
        val reconciler = PaymongoCheckoutReconciler(
            PaymongoProperties(), client, Mockito.mock(PaymongoWebhookService::class.java), payments, orders, testTransactionTemplate(),
        )

        reconciler.expireOrder(order.id)

        assertEquals(PaymentStatus.FAILED, payment.status)
        assertEquals(OrderStatus.EXPIRED, order.status)
        Mockito.verifyNoInteractions(client)
    }

    @Test
    fun `reconcileOrder is a no-op without a pending intent`() {
        val client = Mockito.mock(PaymongoClient::class.java)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val orderId = UUID.randomUUID()
        Mockito.`when`(payments.findByOrderId(orderId)).thenReturn(emptyList())
        val reconciler = PaymongoCheckoutReconciler(
            PaymongoProperties(),
            client,
            Mockito.mock(PaymongoWebhookService::class.java),
            payments,
            Mockito.mock(OrderRepository::class.java),
            testTransactionTemplate(),
        )

        reconciler.reconcileOrder(orderId)

        Mockito.verifyNoInteractions(client)
    }

    private fun qrphOrder() = Order(
        eventId = UUID.randomUUID(),
        recipientEmail = "runner@example.com",
        paymentMethodWire = PaymentMethod.QRPH.wire,
        totalPhp = BigDecimal("125.00"),
    )

    private fun intent(status: String) = PaymongoPaymentIntentResponse(
        PaymongoPaymentIntentResponseEnvelope(
            id = "pi_test",
            attributes = PaymongoPaymentIntentResponseAttributes(
                status = status,
                payments = if (status == "succeeded") listOf(PaymongoPaymentResource(id = "pay_test")) else emptyList(),
            ),
        ),
    )

    private fun checkout(status: String) = PaymongoCheckoutSessionResponse(
        PaymongoCheckoutSessionResponseEnvelope(
            id = "cs_test",
            attributes = PaymongoCheckoutSessionResponseAttributes(status = status),
        ),
    )

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
