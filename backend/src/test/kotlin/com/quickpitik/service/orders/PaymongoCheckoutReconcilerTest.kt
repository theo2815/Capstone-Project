package com.quickpitik.service.orders

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponse
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseAttributes
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseEnvelope
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponse
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponseAttributes
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponseEnvelope
import com.quickpitik.dto.orders.PaymongoPaymentAttributes
import com.quickpitik.dto.orders.PaymongoPaymentResource
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Payment
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.repository.OrderItemRepository
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
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

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
            testTransactionTemplate(), Mockito.mock(OrderItemRepository::class.java),
        )

        reconciler.reconcile()

        assertFalse(providerCallWasTransactional)
        assertEquals(PaymentStatus.FAILED, payment.status)
        assertEquals(OrderStatus.EXPIRED, order.status)
        Mockito.verify(client).expireCheckoutSession("cs_test")
        Mockito.verifyNoInteractions(webhooks)
    }

    @Test
    fun `stale checkout paid while expiring is settled instead of locally expired`() {
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
        Mockito.`when`(client.retrieveCheckoutSession("cs_test")).thenReturn(checkout("active"))
        Mockito.`when`(client.expireCheckoutSession("cs_test")).thenReturn(
            PaymongoCheckoutSessionResponse(PaymongoCheckoutSessionResponseEnvelope(
                id = "cs_test",
                attributes = PaymongoCheckoutSessionResponseAttributes(
                    status = "expired",
                    payments = listOf(PaymongoPaymentResource(
                        id = "pay_late",
                        attributes = com.quickpitik.dto.orders.PaymongoPaymentAttributes(status = "paid"),
                    )),
                ),
            )),
        )
        val reconciler = PaymongoCheckoutReconciler(
            PaymongoProperties(),
            client,
            webhooks,
            payments,
            orders,
            testTransactionTemplate(), Mockito.mock(OrderItemRepository::class.java),
        )

        reconciler.reconcile()

        assertEquals(PaymentStatus.PENDING, payment.status)
        assertEquals(OrderStatus.PENDING, order.status)
        Mockito.verify(webhooks).settleCheckoutSession("cs_test", "pay_late")
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
            intent("succeeded", failedThenPaid())
        }
        val reconciler = PaymongoCheckoutReconciler(PaymongoProperties(), client, webhooks, payments, orders, testTransactionTemplate(), Mockito.mock(OrderItemRepository::class.java))

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
        val reconciler = PaymongoCheckoutReconciler(PaymongoProperties(), client, webhooks, payments, orders, testTransactionTemplate(), Mockito.mock(OrderItemRepository::class.java))

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
        val reconciler = PaymongoCheckoutReconciler(PaymongoProperties(), client, webhooks, payments, orders, testTransactionTemplate(), Mockito.mock(OrderItemRepository::class.java))

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
            PaymongoProperties(), client, Mockito.mock(PaymongoWebhookService::class.java), payments, orders, testTransactionTemplate(), Mockito.mock(OrderItemRepository::class.java),
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
            testTransactionTemplate(), Mockito.mock(OrderItemRepository::class.java),
        )

        reconciler.reconcileOrder(orderId)

        Mockito.verifyNoInteractions(client)
    }

    @Test
    fun `purge closes an unpaid intent before releasing expired order items`() {
        val fixture = expiredReservation("pi_test")
        Mockito.`when`(fixture.client.retrievePaymentIntent("pi_test")).thenReturn(intent("awaiting_next_action"))
        Mockito.`when`(fixture.client.cancelPaymentIntent("pi_test")).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            intent("canceled")
        }

        assertTrue(fixture.reconciler.releaseExpiredPhotoReservations(fixture.photoId))

        val sequence = Mockito.inOrder(fixture.client, fixture.items)
        sequence.verify(fixture.client).cancelPaymentIntent("pi_test")
        sequence.verify(fixture.items).deleteUnpaidReferences(fixture.photoId, listOf(fixture.order.id))
    }

    @Test
    fun `purge also closes the payment intent behind an expired checkout page`() {
        val fixture = expiredReservation("cs_test")
        Mockito.`when`(fixture.client.retrieveCheckoutSession("cs_test")).thenReturn(checkout("active"))
        Mockito.`when`(fixture.client.expireCheckoutSession("cs_test")).thenReturn(
            PaymongoCheckoutSessionResponse(PaymongoCheckoutSessionResponseEnvelope(
                id = "cs_test",
                attributes = PaymongoCheckoutSessionResponseAttributes(
                    status = "expired", paymentIntent = intent("awaiting_next_action").data,
                ),
            )),
        )
        Mockito.`when`(fixture.client.retrievePaymentIntent("pi_test")).thenReturn(intent("awaiting_next_action"))
        Mockito.`when`(fixture.client.cancelPaymentIntent("pi_test")).thenReturn(intent("cancelled"))

        assertTrue(fixture.reconciler.releaseExpiredPhotoReservations(fixture.photoId))

        Mockito.verify(fixture.client).cancelPaymentIntent("pi_test")
        Mockito.verify(fixture.items).deleteUnpaidReferences(fixture.photoId, listOf(fixture.order.id))
    }

    @Test
    fun `purge preserves expired reservations when provider closure is not proven`() {
        for (status in listOf("awaiting_next_action", "processing", "succeeded")) {
            val fixture = expiredReservation("pi_test")
            Mockito.`when`(fixture.client.retrievePaymentIntent("pi_test")).thenReturn(intent("awaiting_next_action"))
            Mockito.`when`(fixture.client.cancelPaymentIntent("pi_test")).thenReturn(intent(status))

            assertFalse(fixture.reconciler.releaseExpiredPhotoReservations(fixture.photoId))
            Mockito.verifyNoInteractions(fixture.items)
        }
        val fixture = expiredReservation("pi_test")
        Mockito.`when`(fixture.client.retrievePaymentIntent("pi_test")).thenThrow(IllegalStateException("provider unavailable"))
        assertFailsWith<IllegalStateException> { fixture.reconciler.releaseExpiredPhotoReservations(fixture.photoId) }
        Mockito.verifyNoInteractions(fixture.items)
    }

    @Test
    fun `purge hands a late successful payment to settlement and keeps its photos`() {
        val fixture = expiredReservation("pi_test")
        Mockito.`when`(fixture.client.retrievePaymentIntent("pi_test")).thenReturn(intent("succeeded", failedThenPaid()))

        assertFalse(fixture.reconciler.releaseExpiredPhotoReservations(fixture.photoId))

        Mockito.verify(fixture.webhooks).settleCheckoutSession(eqArg("pi_test"), eqArg("pay_test"), anyArg())
        Mockito.verify(fixture.client, Mockito.never()).cancelPaymentIntent(anyArg())
        Mockito.verifyNoInteractions(fixture.items)
    }

    @Test
    fun `purge rechecks payment and order under locks after provider IO`() {
        val fixture = expiredReservation("pi_test")
        Mockito.`when`(fixture.client.retrievePaymentIntent("pi_test")).thenAnswer {
            fixture.order.status = OrderStatus.FULFILLED
            fixture.payment.status = PaymentStatus.SUCCEEDED
            intent("canceled")
        }

        assertFalse(fixture.reconciler.releaseExpiredPhotoReservations(fixture.photoId))
        Mockito.verifyNoInteractions(fixture.items)
    }

    @Test
    fun `purge keeps reservations with a pending order or unknown provider outcome`() {
        for (ref in listOf(null, "", "unknown_ref", "pi_test")) {
            val fixture = expiredReservation(ref)
            if (ref == "pi_test") fixture.order.status = OrderStatus.PENDING
            assertFalse(fixture.reconciler.releaseExpiredPhotoReservations(fixture.photoId))
            Mockito.verifyNoInteractions(fixture.client, fixture.items)
        }
    }

    private class ReservationFixture(
        val photoId: UUID,
        val order: Order,
        val payment: Payment,
        val client: PaymongoClient,
        val webhooks: PaymongoWebhookService,
        val items: OrderItemRepository,
        val reconciler: PaymongoCheckoutReconciler,
    )

    private fun expiredReservation(ref: String?): ReservationFixture {
        val photoId = UUID.randomUUID()
        val order = qrphOrder().apply { status = OrderStatus.EXPIRED }
        val payment = Payment(orderId = order.id, provider = "paymongo", providerRef = ref, amountPhp = order.totalPhp,
            status = PaymentStatus.FAILED)
        val client = Mockito.mock(PaymongoClient::class.java)
        val payments = Mockito.mock(PaymentRepository::class.java)
        val orders = Mockito.mock(OrderRepository::class.java)
        val items = Mockito.mock(OrderItemRepository::class.java)
        val webhooks = Mockito.mock(PaymongoWebhookService::class.java)
        Mockito.`when`(orders.findReferencingPhoto(photoId)).thenReturn(listOf(order))
        Mockito.`when`(orders.findByIdForUpdate(order.id)).thenReturn(order)
        Mockito.`when`(payments.findByOrderId(order.id)).thenReturn(listOf(payment))
        Mockito.`when`(payments.findAllByOrderIdInForUpdate(listOf(order.id))).thenReturn(listOf(payment))
        return ReservationFixture(photoId, order, payment, client, webhooks, items,
            PaymongoCheckoutReconciler(PaymongoProperties(), client, webhooks, payments, orders, testTransactionTemplate(), items))
    }

    private fun qrphOrder() = Order(
        eventId = UUID.randomUUID(),
        recipientEmail = "runner@example.com",
        paymentMethodWire = PaymentMethod.QRPH.wire,
        totalPhp = BigDecimal("125.00"),
    )

    private fun intent(
        status: String,
        payments: List<PaymongoPaymentResource> = if (status == "succeeded") {
            listOf(PaymongoPaymentResource(id = "pay_test", attributes = PaymongoPaymentAttributes(status = "paid")))
        } else {
            emptyList()
        },
    ) = PaymongoPaymentIntentResponse(
        PaymongoPaymentIntentResponseEnvelope(
            id = "pi_test",
            attributes = PaymongoPaymentIntentResponseAttributes(
                status = status,
                payments = payments,
            ),
        ),
    )

    private fun failedThenPaid() = listOf(
        PaymongoPaymentResource(id = "pay_failed", attributes = PaymongoPaymentAttributes(status = "failed")),
        PaymongoPaymentResource(id = "pay_test", attributes = PaymongoPaymentAttributes(status = "paid")),
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
