package com.quickpitik.service.orders

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.PaymentRepository
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.time.OffsetDateTime
import java.util.UUID

@Service
class PaymongoCheckoutReconciler(
    private val properties: PaymongoProperties,
    private val paymongoClient: PaymongoClient,
    private val webhookService: PaymongoWebhookService,
    private val paymentRepository: PaymentRepository,
    private val orderRepository: OrderRepository,
    private val transactionTemplate: TransactionTemplate,
    private val orderItemRepository: OrderItemRepository,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Scheduled(fixedDelayString = "\${app.payments.paymongo.reconcile-interval-ms:60000}")
    fun reconcile() {
        paymentRepository.findByProviderAndStatusAndProviderRefStartingWithOrderByCreatedAtAsc(
            "paymongo",
            PaymentStatus.PENDING,
            "pi_",
            PageRequest.of(0, BATCH_SIZE),
        ).groupBy { it.providerRef!! }.values.forEach(::reconcilePaymentIntent)

        val cutoff = OffsetDateTime.now().minus(properties.checkoutTtl)
        // ponytail: a missing providerRef means the provider outcome is
        // unknown; keep the photos reserved for same-key retry/manual repair.
        // Add a durable checkout outbox if this becomes operationally common.
        val stale = paymentRepository.findByProviderAndStatusAndProviderRefIsNotNullAndCreatedAtBeforeOrderByCreatedAtAsc(
            "paymongo",
            PaymentStatus.PENDING,
            cutoff,
            PageRequest.of(0, BATCH_SIZE),
        )
        stale.filterNot { it.providerRef!!.startsWith("pi_") }
            .groupBy { it.providerRef!! }
            .values
            .forEach(::reconcileGroup)
    }

    // On-demand variant of the QRPH pass for one order: the checkout's
    // "I've paid" check asks PayMongo now instead of waiting for the webhook
    // or the next sweep. Reconciles every sibling row on the same intent so a
    // multi-event split settles or expires together.
    fun reconcileOrder(orderId: UUID) {
        val intentId = paymentRepository.findByOrderId(orderId)
            .firstOrNull { it.status == PaymentStatus.PENDING && it.providerRef?.startsWith("pi_") == true }
            ?.providerRef ?: return
        val pending = paymentRepository.findAllByProviderAndProviderRef("paymongo", intentId)
            .filter { it.status == PaymentStatus.PENDING }
        if (pending.isNotEmpty()) reconcilePaymentIntent(pending)
    }

    // Runner-initiated cancel: same terminal state as a timeout.
    fun expireOrder(orderId: UUID) = markExpired(listOf(orderId))

    // Permanent photo deletion may release an abandoned checkout only after
    // its provider payment is conclusively closed. Local EXPIRED alone is not
    // proof: a late paid webhook must still be allowed to fulfil the order.
    // Provider I/O happens before the short payment/order lock transaction.
    fun releaseExpiredPhotoReservations(photoId: UUID): Boolean {
        val orders = orderRepository.findReferencingPhoto(photoId)
        if (orders.isEmpty()) return true
        if (orders.any { it.status != OrderStatus.EXPIRED || it.paidAt != null }) return false
        val orderIds = orders.map { it.id }
        val payments = orderIds.flatMap(paymentRepository::findByOrderId)
        if (payments.any { it.status == PaymentStatus.SUCCEEDED || it.paidAt != null ||
                it.provider != "paymongo" || it.providerRef.isNullOrBlank() }) return false
        val refs = payments.map { it.providerRef!! }.distinct()
        if (!refs.all(::closeUnpaidPayment)) return false

        return transactionTemplate.execute {
            // Same lock order as fulfillment; recheck after any late webhook.
            val lockedPayments = paymentRepository.findAllByOrderIdInForUpdate(orderIds)
            val lockedOrders = orderIds.mapNotNull(orderRepository::findByIdForUpdate)
            if (lockedOrders.size != orderIds.size || lockedOrders.any {
                    it.status != OrderStatus.EXPIRED || it.paidAt != null
                } || lockedPayments.any {
                    it.status == PaymentStatus.SUCCEEDED || it.paidAt != null || it.providerRef !in refs
                } || lockedPayments.map { it.id }.toSet() != payments.map { it.id }.toSet()) {
                false
            } else {
                orderItemRepository.deleteUnpaidReferences(photoId, orderIds)
                true
            }
        } == true
    }

    private fun closeUnpaidPayment(ref: String): Boolean {
        if (ref.startsWith("pi_")) return closeUnpaidIntent(ref, ref)
        if (!ref.startsWith("cs_")) return false
        var checkout = paymongoClient.retrieveCheckoutSession(ref)
        checkout.data.attributes.payments.firstOrNull { it.attributes.status == "paid" }?.let {
            webhookService.settleCheckoutSession(ref, it.id)
            return false
        }
        if (checkout.data.attributes.status != "expired") {
            checkout = paymongoClient.expireCheckoutSession(ref)
        }
        val paid = checkout.data.attributes.payments.firstOrNull { it.attributes.status == "paid" }
        if (paid != null) {
            webhookService.settleCheckoutSession(ref, paid.id)
            return false
        }
        if (checkout.data.attributes.status != "expired") return false
        // Expiring the page stops new attempts; also close an existing intent
        // so a payment already opened in another tab cannot settle after purge.
        return checkout.data.attributes.paymentIntent?.id?.let { closeUnpaidIntent(it, ref) } ?: true
    }

    private fun closeUnpaidIntent(intentId: String, paymentRef: String): Boolean {
        var intent = paymongoClient.retrievePaymentIntent(intentId)
        if (intent.data.attributes.status == "succeeded") {
            webhookService.settleCheckoutSession(paymentRef, intent.data.attributes.payments.firstOrNull()?.id)
            return false
        }
        if (intent.data.attributes.status !in setOf("cancelled", "canceled")) {
            intent = paymongoClient.cancelPaymentIntent(intentId)
        }
        return intent.data.attributes.status in setOf("cancelled", "canceled") &&
            intent.data.attributes.payments.none { it.attributes.status == "paid" }
    }

    private fun reconcilePaymentIntent(payments: List<com.quickpitik.entity.Payment>) {
        val intentId = payments.first().providerRef!!
        val intent = try {
            paymongoClient.retrievePaymentIntent(intentId)
        } catch (ex: Exception) {
            log.warn("Payment Intent reconciliation deferred for {}: {}", intentId, ex.message)
            return
        }
        if (intent.data.attributes.status == "succeeded") {
            webhookService.settlePaymentIntent(
                intentId,
                intent.data.attributes.payments.firstOrNull()?.id,
                intent.data.attributes.metadata,
            )
            return
        }
        // A declined / simulator-failed payment drops the intent back to
        // awaiting_payment_method with last_payment_error set; the QR it was
        // attached to is spent. Treat it like expiry so the runner is told now
        // instead of at the QR's deadline.
        val attrs = intent.data.attributes
        val failed = attrs.status == "awaiting_payment_method" && attrs.lastPaymentError != null
        if (failed || payments.mapNotNull { it.expiresAt }.minOrNull()?.isBefore(OffsetDateTime.now()) == true) {
            markExpired(payments.map { it.orderId })
        }
    }

    private fun reconcileGroup(payments: List<com.quickpitik.entity.Payment>) {
        val sessionId = payments.first().providerRef!!

        var checkout = try {
            paymongoClient.retrieveCheckoutSession(sessionId)
        } catch (ex: Exception) {
            log.warn("Checkout reconciliation deferred for {}: {}", sessionId, ex.message)
            return
        }
        val paid = checkout.data.attributes.payments.firstOrNull {
            it.attributes.status.equals("paid", ignoreCase = true)
        }
        if (paid != null) {
            webhookService.settleCheckoutSession(sessionId, paid.id)
            return
        }

        if (!checkout.data.attributes.status.equals("expired", ignoreCase = true)) {
            try {
                checkout = paymongoClient.expireCheckoutSession(sessionId)
            } catch (ex: Exception) {
                log.warn("Checkout expiry deferred for {}: {}", sessionId, ex.message)
                return
            }
        }
        val finalPaid = checkout.data.attributes.payments.firstOrNull {
            it.attributes.status.equals("paid", ignoreCase = true)
        }
        if (finalPaid != null) {
            webhookService.settleCheckoutSession(sessionId, finalPaid.id)
            return
        }
        markExpired(payments.map { it.orderId })
    }

    private fun markExpired(orderIds: Collection<UUID>) {
        transactionTemplate.executeWithoutResult {
            paymentRepository.findAllByOrderIdInForUpdate(orderIds).forEach { payment ->
                if (payment.status == PaymentStatus.PENDING) {
                    payment.status = PaymentStatus.FAILED
                    paymentRepository.save(payment)
                }
                val order = orderRepository.findByIdForUpdate(payment.orderId)
                if (order?.status == OrderStatus.PENDING) {
                    order.status = OrderStatus.EXPIRED
                    orderRepository.save(order)
                }
            }
        }
    }

    private companion object {
        const val BATCH_SIZE = 100
    }
}
