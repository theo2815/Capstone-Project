package com.quickpitik.service.orders

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.repository.OrderRepository
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
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Scheduled(fixedDelayString = "\${app.payments.paymongo.reconcile-interval-ms:60000}")
    fun reconcile() {
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
        stale.groupBy { it.providerRef!! }.values.forEach(::reconcileGroup)
    }

    private fun reconcileGroup(payments: List<com.quickpitik.entity.Payment>) {
        val sessionId = payments.first().providerRef!!

        val checkout = try {
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
                paymongoClient.expireCheckoutSession(sessionId)
            } catch (ex: Exception) {
                log.warn("Checkout expiry deferred for {}: {}", sessionId, ex.message)
                return
            }
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
