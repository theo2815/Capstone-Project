package com.quickpitik.service.orders

import com.quickpitik.dto.orders.PaymongoWebhookEvent
import com.quickpitik.entity.DownloadGrant
import com.quickpitik.entity.DownloadGrantId
import com.quickpitik.entity.OrderStatus
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

// Dispatches verified PayMongo webhook events to the right handler. Idempotent
// by design: each `checkout_session.payment.paid` event re-applied is a no-op
// once the matching orders are already PAID (status guard) and grants are
// minted (UNIQUE on download_grants).
//
// Multi-event carts share one Checkout Session id (cs_…) across N Payment rows
// — this handler flips them all in one TX so the user's /orders view stays
// consistent post-redirect.
@Service
@Transactional
class PaymongoWebhookService(
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val paymentRepository: PaymentRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val transactionMintingService: TransactionMintingService,
    private val eventPublisher: ApplicationEventPublisher,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun handle(event: PaymongoWebhookEvent): Map<String, Any?> {
        val eventType = event.data.attributes.type
        val resourceId = event.data.attributes.data.id

        return when (eventType) {
            "checkout_session.payment.paid" -> handleCheckoutPaid(resourceId)
            else -> {
                log.info("PayMongo event ignored · type={} resourceId={}", eventType, resourceId)
                mapOf("acknowledged" to true, "applied" to false, "type" to eventType)
            }
        }
    }

    private fun handleCheckoutPaid(checkoutSessionId: String): Map<String, Any?> {
        if (checkoutSessionId.isBlank()) {
            log.warn("PayMongo checkout_session.payment.paid missing data.id — ignoring")
            return mapOf("acknowledged" to true, "applied" to false, "reason" to "missing cs id")
        }

        // Every Payment row carrying this cs_id maps to one Order in the
        // multi-event split. Single-event carts return exactly one row;
        // multi-event carts return N.
        val payments = paymentRepository.findAllByProviderAndProviderRef("paymongo", checkoutSessionId)
        if (payments.isEmpty()) {
            log.warn("PayMongo cs={} has no matching Payment rows — ignoring", checkoutSessionId)
            return mapOf("acknowledged" to true, "applied" to false, "reason" to "no payments")
        }

        val now = OffsetDateTime.now()
        var ordersPaid = 0
        var grantsMinted = 0
        val justFlipped = mutableListOf<UUID>()

        for (payment in payments) {
            if (payment.status != PaymentStatus.SUCCEEDED) {
                payment.status = PaymentStatus.SUCCEEDED
                payment.paidAt = now
                paymentRepository.save(payment)
            }
            val order = orderRepository.findById(payment.orderId).orElse(null) ?: continue
            if (order.status == OrderStatus.PAID || order.status == OrderStatus.FULFILLED) {
                continue
            }
            order.status = OrderStatus.PAID
            order.paidAt = now
            orderRepository.save(order)
            ordersPaid += 1
            grantsMinted += mintGrantsIfMissing(order.id, now)
            transactionMintingService.mintForPaidOrder(order.id)
            justFlipped.add(order.id)
        }

        // Publish AFTER all DB writes in this TX so the AFTER_COMMIT listener
        // sees the persisted state. The listener is @Async and idempotent on
        // orders.email_sent_at, so re-deliveries of the same webhook won't
        // double-send.
        justFlipped.forEach { orderId ->
            eventPublisher.publishEvent(
                OrderPaidEvent(orderId = orderId, checkoutSessionId = checkoutSessionId),
            )
        }

        log.info(
            "PayMongo cs={} applied · ordersPaid={} grantsMinted={}",
            checkoutSessionId,
            ordersPaid,
            grantsMinted,
        )
        return mapOf(
            "acknowledged" to true,
            "applied" to true,
            "checkoutSessionId" to checkoutSessionId,
            "ordersPaid" to ordersPaid,
            "grantsMinted" to grantsMinted,
        )
    }

    private fun mintGrantsIfMissing(orderId: UUID, now: OffsetDateTime): Int {
        val existing = downloadGrantRepository.findByIdOrderId(orderId).map { it.id.photoId }.toSet()
        val photos = orderItemRepository.findByIdOrderId(orderId).map { it.id.photoId }
        val grantedUntil = now.plusYears(1)
        var minted = 0
        photos.filter { it !in existing }.forEach { photoId ->
            downloadGrantRepository.save(
                DownloadGrant(
                    id = DownloadGrantId(orderId = orderId, photoId = photoId),
                    grantedUntil = grantedUntil,
                ),
            )
            minted += 1
        }
        return minted
    }
}
