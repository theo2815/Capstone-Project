package com.quickpitik.service.earnings

import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.earnings.privacyBuyerDisplay
import com.quickpitik.dto.earnings.privacyGuestBuyerDisplay
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeResolution
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
 * the earnings rollups have a single source of truth (Q-E3). Refund rows
 * are minted here too, by mintRefund() — one writer for the whole ledger.
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
                .orElse(privacyGuestBuyerDisplay(order.recipientEmail))
        } ?: privacyGuestBuyerDisplay(order.recipientEmail)

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
                eventPhotographerRepository.adjustSalesCounters(
                    order.eventId,
                    photographerId,
                    saleDelta = 1,
                    keptDelta = amountKept,
                )
            } catch (ex: DataIntegrityViolationException) {
                // Concurrent mint hit the unique index — clean no-op.
                log.debug("transaction already exists for order={} photo={}", orderId, item.id.photoId)
            }
        }
    }

    /**
     * Reverses one sale row when a dispute refund settles. Same owner as the
     * sale side so the ledger keeps a single writer; the unique index on
     * (order_id, photo_id, is_refund) is the idempotency.
     */
    @Transactional(propagation = Propagation.REQUIRED)
    fun mintRefund(dispute: Dispute, amount: BigDecimal, photographerId: UUID) {
        val existing = transactionRepository.findByOrderIdAndPhotoIdAndIsRefund(
            dispute.orderId,
            dispute.photoId,
            true,
        )
        if (existing != null) return
        val original = transactionRepository.findByOrderIdAndPhotoIdAndIsRefund(
            dispute.orderId,
            dispute.photoId,
            false,
        )
        val order = orderRepository.findById(dispute.orderId).orElse(null) ?: return
        val (keptReversal, discountReversal) = refundLedgerSplit(dispute, original, amount)
        try {
            transactionRepository.save(
                Transaction(
                    paidAt = order.paidAt ?: order.createdAt,
                    photographerId = photographerId,
                    eventId = order.eventId,
                    photoId = dispute.photoId,
                    orderId = order.id,
                    buyerId = order.userId,
                    buyerDisplayName = original?.buyerDisplayName ?: "",
                    amountKeptPhp = keptReversal,
                    discountPhp = discountReversal,
                    isRefund = true,
                    refundOf = original?.id,
                ),
            )
            eventPhotographerRepository.adjustSalesCounters(
                order.eventId,
                photographerId,
                saleDelta = if (dispute.resolution == DisputeResolution.REFUND_FULL) -1 else 0,
                keptDelta = keptReversal,
            )
        } catch (_: DataIntegrityViolationException) {
            // Concurrent reconciliation already minted the unique refund row.
        }
    }

    // What comes back off the ledger, as (kept, discount) negatives. With the
    // original sale row in hand the reversal is exact: a full refund negates
    // it outright, a partial refund scales both figures by the fraction of the
    // charged price returned. Rows without a sale row (pre-V9 legacy) keep the
    // old keepRate estimate.
    private fun refundLedgerSplit(
        dispute: Dispute,
        original: Transaction?,
        amount: BigDecimal,
    ): Pair<BigDecimal, BigDecimal> {
        val legacy = amount.multiply(platformProperties.photographerKeepRate)
            .setScale(2, RoundingMode.HALF_UP)
            .negate() to BigDecimal.ZERO
        if (original == null) return legacy
        if (dispute.resolution == DisputeResolution.REFUND_FULL) {
            return original.amountKeptPhp.negate() to original.discountPhp.negate()
        }
        val charged = orderItemRepository.findByIdOrderId(dispute.orderId)
            .firstOrNull { it.id.photoId == dispute.photoId }
            ?.let { it.pricePhpAtPurchase.subtract(it.discountPhp) }
            ?.takeIf { it.signum() > 0 }
            ?: return legacy
        val fraction = amount.divide(charged, 10, RoundingMode.HALF_UP)
        return original.amountKeptPhp.multiply(fraction).setScale(2, RoundingMode.HALF_UP).negate() to
            original.discountPhp.multiply(fraction).setScale(2, RoundingMode.HALF_UP).negate()
    }

}
