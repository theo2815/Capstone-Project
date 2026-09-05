package com.quickpitik.service.earnings

import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.earnings.privacyBuyerDisplay
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.Transaction
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import org.slf4j.LoggerFactory
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.math.RoundingMode
import java.util.UUID

/**
 * Mints one `transactions` row per order_item the moment an order reaches PAID,
 * keying off the photo's photographer_id. Splits PR 5's payment side from the
 * earnings ledger so OrderService stays focused on cart → order → grants and
 * the earnings rollups have a single source of truth (Q-E3 — refunds will be
 * minted from the same service later by admin's dispute-resolve flow).
 *
 * Idempotency: the unique index on (order_id, photo_id, is_refund) means a
 * second mint pass for the same paid order is a clean no-op. Useful when
 * PaymentWebhookService re-fires for an order that's already been minted via
 * OrderService.create.
 *
 * Counters: bumps `event_photographer.sales_count` and `revenue_kept_php` so
 * the per-event earnings tile reads from the same denormalised counters that
 * already drive PR 7's photo_count tile — one query path per dashboard view.
 */
@Service
class TransactionMintingService(
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val photoRepository: PhotoRepository,
    private val userRepository: UserRepository,
    private val transactionRepository: TransactionRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val platformProperties: PlatformProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional(propagation = Propagation.REQUIRED)
    fun mintForPaidOrder(orderId: UUID) {
        val order = orderRepository.findById(orderId).orElse(null) ?: return
        val paidAt = order.paidAt ?: order.createdAt
        val items = orderItemRepository.findByIdOrderId(orderId)
        if (items.isEmpty()) return
        val photos = photoRepository.findAllById(items.map { it.id.photoId }).associateBy { it.id }

        // Resolve buyer display once per order — every row attributes to the
        // same person so we don't need to re-query per item.
        val buyerDisplay = order.userId?.let { uid ->
            userRepository.findById(uid)
                .map { privacyBuyerDisplay(it.name) }
                .orElse(displayFromEmail(order.recipientEmail))
        } ?: displayFromEmail(order.recipientEmail)

        val keepRate = platformProperties.photographerKeepRate

        for (item in items) {
            val photo = photos[item.id.photoId] ?: continue
            val photographerId = photo.photographerId ?: continue

            // Skip if a sale row already exists for this (order, photo).
            val existing = transactionRepository.findByOrderIdAndPhotoIdAndIsRefund(
                orderId = orderId,
                photoId = item.id.photoId,
                isRefund = false,
            )
            if (existing != null) continue
            // A 100% giveaway charged nothing: no share, no cut, no sale row.
            if (item.pricePhpAtPurchase.subtract(item.discountPhp).signum() == 0) continue

            // A photographer coupon comes out of the photographer's share only:
            // the platform's cut (list price × cutRate) is identical with or
            // without a code. See couponDiscount() in CouponService.
            val amountKept = item.pricePhpAtPurchase
                .multiply(keepRate)
                .setScale(2, RoundingMode.HALF_UP)
                .subtract(item.discountPhp)

            try {
                transactionRepository.save(
                    Transaction(
                        paidAt = paidAt,
                        photographerId = photographerId,
                        eventId = order.eventId,
                        photoId = item.id.photoId,
                        orderId = orderId,
                        buyerId = order.userId,
                        buyerDisplayName = buyerDisplay,
                        amountKeptPhp = amountKept,
                        discountPhp = item.discountPhp,
                        isRefund = false,
                    ),
                )
                bumpEventPhotographer(order.eventId, photographerId, amountKept)
            } catch (ex: DataIntegrityViolationException) {
                // Concurrent mint hit the unique index — clean no-op.
                log.debug("transaction already exists for order={} photo={}", orderId, item.id.photoId)
            }
        }
    }

    private fun bumpEventPhotographer(eventId: UUID, photographerId: UUID, amount: BigDecimal) {
        val key = EventPhotographerId(eventId = eventId, photographerId = photographerId)
        val ep = eventPhotographerRepository.findById(key)
            .orElseGet {
                // PR 7 normally creates this row on first upload; lazily create
                // it here too in case a sale lands before any upload was tracked
                // (covered events where rows were imported by hand for testing).
                EventPhotographer(id = key)
            }
        ep.salesCount = ep.salesCount + 1
        ep.revenueKeptPhp = ep.revenueKeptPhp.add(amount)
        eventPhotographerRepository.save(ep)
    }

    private fun displayFromEmail(email: String): String {
        if (email.isBlank()) return ""
        // Guest checkout: use the local part of the email as a stable initial
        // (e.g., "aira.santos@gmail.com" → "Aira S."). Falls back to the raw
        // local part when there is no separator.
        val local = email.substringBefore('@').trim()
        if (local.isEmpty()) return ""
        val parts = local.split(Regex("[._\\-]")).filter { it.isNotBlank() }
        if (parts.isEmpty()) return local.replaceFirstChar { it.uppercase() }
        if (parts.size == 1) return parts[0].replaceFirstChar { it.uppercase() }
        val first = parts[0].replaceFirstChar { it.uppercase() }
        val lastInitial = parts.last().firstOrNull()?.uppercase() ?: return first
        return "$first $lastInitial."
    }
}
