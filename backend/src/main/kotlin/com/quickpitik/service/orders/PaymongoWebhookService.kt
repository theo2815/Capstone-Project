package com.quickpitik.service.orders

import com.quickpitik.dto.orders.PaymongoWebhookEvent
import com.quickpitik.entity.DownloadGrant
import com.quickpitik.entity.DownloadGrantId
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Payment
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.service.earnings.TransactionMintingService
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

@Service
class PaymongoWebhookService(
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val paymentRepository: PaymentRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val transactionMintingService: TransactionMintingService,
    private val eventPublisher: ApplicationEventPublisher,
    private val refundService: PaymongoRefundService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional
    fun handle(event: PaymongoWebhookEvent): Map<String, Any?> {
        val type = event.data.attributes.type
        val resource = event.data.attributes.data
        return when (type) {
            "checkout_session.payment.paid" -> settleCheckoutSession(
                checkoutSessionId = resource.id,
                providerPaymentId = resource.attributes.payments
                    .firstOrNull { it.attributes.status.equals("paid", ignoreCase = true) }
                    ?.id
                    ?: resource.attributes.payments.firstOrNull()?.id,
                metadata = resource.attributes.metadata,
            )
            "payment.refunded", "payment.refund.updated", "refund.succeeded" -> {
                val applied = if (resource.id.startsWith("ref_")) {
                    refundService.handleWebhook(resource.id, resource.attributes.status ?: "succeeded")
                } else {
                    false
                }
                mapOf("acknowledged" to true, "applied" to applied, "type" to type)
            }
            else -> {
                log.info("PayMongo event ignored - type={} resourceId={}", type, resource.id)
                mapOf("acknowledged" to true, "applied" to false, "type" to type)
            }
        }
    }

    /** Also used by the stale-checkout reconciler when a paid webhook was missed. */
    @Transactional
    fun settleCheckoutSession(
        checkoutSessionId: String,
        providerPaymentId: String?,
        metadata: Map<String, String>? = null,
    ): Map<String, Any?> {
        if (checkoutSessionId.isBlank()) {
            return mapOf("acknowledged" to true, "applied" to false, "reason" to "missing cs id")
        }

        val payments = lockedPayments(checkoutSessionId, metadata)
        if (payments.isEmpty()) {
            log.warn("PayMongo cs={} has no matching Payment rows", checkoutSessionId)
            return mapOf("acknowledged" to true, "applied" to false, "reason" to "no payments")
        }

        val now = OffsetDateTime.now()
        var ordersFulfilled = 0
        var grantsMinted = 0
        val justFulfilled = mutableListOf<UUID>()

        payments.forEach { payment ->
            payment.providerRef = checkoutSessionId
            providerPaymentId?.takeIf { it.isNotBlank() }?.let { payment.providerPaymentId = it }
            payment.status = PaymentStatus.SUCCEEDED
            payment.paidAt = payment.paidAt ?: now
            paymentRepository.save(payment)

            val order = orderRepository.findByIdForUpdate(payment.orderId) ?: return@forEach
            if (order.status == OrderStatus.FULFILLED || order.status == OrderStatus.REFUNDED) return@forEach

            order.paidAt = order.paidAt ?: now
            grantsMinted += mintGrantsIfMissing(order.id, now)
            transactionMintingService.mintForPaidOrder(order.id)
            order.status = OrderStatus.FULFILLED
            orderRepository.save(order)
            ordersFulfilled++
            justFulfilled.add(order.id)
        }

        justFulfilled.forEach { orderId ->
            eventPublisher.publishEvent(OrderPaidEvent(orderId, checkoutSessionId))
        }
        log.info(
            "PayMongo cs={} applied - ordersFulfilled={} grantsMinted={}",
            checkoutSessionId,
            ordersFulfilled,
            grantsMinted,
        )
        return mapOf(
            "acknowledged" to true,
            "applied" to (ordersFulfilled > 0),
            "checkoutSessionId" to checkoutSessionId,
            "ordersFulfilled" to ordersFulfilled,
            "grantsMinted" to grantsMinted,
        )
    }

    private fun lockedPayments(
        checkoutSessionId: String,
        metadata: Map<String, String>?,
    ): List<Payment> {
        paymentRepository.findAllByProviderAndProviderRefForUpdate(PAYMONGO, checkoutSessionId)
            .takeIf { it.isNotEmpty() }
            ?.let { return it }

        val primaryId = metadata?.get("primaryOrderId")?.let { raw ->
            runCatching { UUID.fromString(raw) }.getOrNull()
        }
            ?: return emptyList()
        val primary = orderRepository.findByIdForUpdate(primaryId) ?: return emptyList()
        val key = primary.idempotencyKey ?: return emptyList()
        val orders = scopedGroup(primary, key)
        return paymentRepository.findAllByOrderIdInForUpdate(orders.map { it.id })
    }

    private fun scopedGroup(primary: Order, key: String): List<Order> =
        primary.userId?.let { orderRepository.findByUserIdAndIdempotencyKey(it, key) }
            ?: orderRepository.findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(
                primary.recipientEmail,
                key,
            )

    private fun mintGrantsIfMissing(orderId: UUID, now: OffsetDateTime): Int {
        val existing = downloadGrantRepository.findByIdOrderId(orderId).map { it.id.photoId }.toSet()
        val grantedUntil = now.plusYears(1)
        return orderItemRepository.findByIdOrderId(orderId)
            .map { it.id.photoId }
            .filter { it !in existing }
            .onEach { photoId ->
                downloadGrantRepository.save(DownloadGrant(DownloadGrantId(orderId, photoId), grantedUntil))
            }
            .size
    }

    private companion object {
        const val PAYMONGO = "paymongo"
    }
}
