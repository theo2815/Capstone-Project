package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.orders.PaymongoRefundRequest
import com.quickpitik.dto.orders.PaymongoRefundRequestAttributes
import com.quickpitik.dto.orders.PaymongoRefundRequestEnvelope
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeResolution
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.DownloadGrantId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.RunnerMessageKind
import com.quickpitik.entity.Transaction
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.runner.RunnerMessagesService
import org.slf4j.LoggerFactory
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.math.RoundingMode
import java.time.Duration
import java.time.OffsetDateTime
import java.util.UUID

@Service
class PaymongoRefundService(
    private val disputeRepository: DisputeRepository,
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val paymentRepository: PaymentRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val photoRepository: PhotoRepository,
    private val transactionRepository: TransactionRepository,
    private val platformProperties: PlatformProperties,
    private val paymongoClient: PaymongoClient,
    private val adminDecisionLogService: AdminDecisionLogService,
    private val runnerMessagesService: RunnerMessagesService,
    private val transactionTemplate: TransactionTemplate,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun request(
        adminId: UUID,
        disputeId: UUID,
        resolution: DisputeResolution,
        requestedAmount: BigDecimal?,
        reason: String?,
    ) {
        require(resolution != DisputeResolution.DENY)
        val prepared = transactionTemplate.execute {
            prepare(adminId, disputeId, resolution, requestedAmount, reason)
        } ?: error("Refund preparation returned null")
        if (!prepared.completed) process(prepared, reportFailure = true)
    }

    fun handleWebhook(providerRefundId: String, status: String): Boolean =
        transactionTemplate.execute {
            val dispute = disputeRepository.findByProviderRefundId(providerRefundId) ?: return@execute false
            val locked = disputeRepository.findByIdForUpdate(dispute.id) ?: return@execute false
            locked.refundStatus = status.lowercase()
            disputeRepository.save(locked)
            if (locked.refundStatus == SUCCEEDED) completeSucceededRefund(locked)
            true
        } ?: false

    @Scheduled(fixedDelayString = "\${app.payments.paymongo.refund-reconcile-interval-ms:60000}")
    fun reconcile() {
        disputeRepository.findByRefundStatusInOrderByRefundRequestedAtAsc(
            IN_FLIGHT_STATUSES,
            PageRequest.of(0, BATCH_SIZE),
        ).forEach { dispute ->
            val prepared = transactionTemplate.execute { snapshot(dispute.id) } ?: return@forEach
            process(prepared, reportFailure = false)
        }
    }

    private fun prepare(
        adminId: UUID,
        disputeId: UUID,
        resolution: DisputeResolution,
        requestedAmount: BigDecimal?,
        reason: String?,
    ): PreparedRefund {
        val dispute = disputeRepository.findByIdForUpdate(disputeId)
            ?: throw NotFoundException("Dispute not found", ErrorCodes.DISPUTE_NOT_FOUND)
        if (dispute.refundStatus == SUCCEEDED && dispute.status == DisputeStatus.RESOLVED) {
            return snapshotOf(dispute, completed = true)
        }
        if (dispute.refundStatus == MANUAL_REVIEW) {
            throw checkoutConflict("Refund outcome requires manual PayMongo reconciliation before another attempt.")
        }
        if (dispute.status == DisputeStatus.DENIED || dispute.status == DisputeStatus.WITHDRAWN) {
            throw checkoutConflict("Cannot refund a ${dispute.status.wire} dispute")
        }
        if (dispute.refundStatus in IN_FLIGHT_STATUSES) return snapshotOf(dispute)

        val item = orderItemRepository.findByIdOrderId(dispute.orderId)
            .firstOrNull { it.id.photoId == dispute.photoId }
            ?: throw NotFoundException("Order item not found", ErrorCodes.ORDER_NOT_FOUND)
        // Refund against what the runner paid, not the list price — a coupon
        // item was charged pricePhpAtPurchase − discountPhp (V45).
        val charged = item.pricePhpAtPurchase.subtract(item.discountPhp)
        val amount = when (resolution) {
            DisputeResolution.REFUND_FULL -> charged
            DisputeResolution.REFUND_PARTIAL -> validatePartialAmount(requestedAmount, charged)
            DisputeResolution.DENY -> error("DENY is not a provider refund")
        }
        val payment = paymentRepository.findByOrderId(dispute.orderId)
            .firstOrNull { it.provider == PAYMONGO && it.status == PaymentStatus.SUCCEEDED }
            ?: throw checkoutConflict("The order has no settled PayMongo payment to refund.")
        val sessionId = payment.providerRef
            ?: throw checkoutConflict("The settled payment has no PayMongo Checkout Session ID.")

        val now = OffsetDateTime.now()
        dispute.resolution = resolution
        dispute.refundAmountPhp = amount
        dispute.refundStatus = REQUESTING
        dispute.refundRequestedAt = now
        dispute.refundRequestedBy = adminId
        dispute.refundReason = reason?.trim()?.takeIf { it.isNotEmpty() }
        dispute.providerRefundId = null
        disputeRepository.save(dispute)
        return PreparedRefund(
            disputeId = dispute.id,
            paymentId = payment.providerPaymentId,
            checkoutSessionId = sessionId,
            amount = amount,
            requestedAt = now,
            providerRefundId = null,
        )
    }

    private fun snapshot(disputeId: UUID): PreparedRefund? {
        val dispute = disputeRepository.findByIdForUpdate(disputeId) ?: return null
        if (dispute.refundStatus !in IN_FLIGHT_STATUSES) return null
        return snapshotOf(dispute)
    }

    private fun snapshotOf(dispute: Dispute, completed: Boolean = false): PreparedRefund {
        val payment = paymentRepository.findByOrderId(dispute.orderId)
            .firstOrNull { it.provider == PAYMONGO }
            ?: throw checkoutConflict("The order has no PayMongo payment.")
        return PreparedRefund(
            disputeId = dispute.id,
            paymentId = payment.providerPaymentId,
            checkoutSessionId = payment.providerRef
                ?: throw checkoutConflict("The payment has no Checkout Session ID."),
            amount = dispute.refundAmountPhp
                ?: throw checkoutConflict("The refund has no amount."),
            requestedAt = dispute.refundRequestedAt ?: dispute.openedAt,
            providerRefundId = dispute.providerRefundId,
            completed = completed,
        )
    }

    private fun process(prepared: PreparedRefund, reportFailure: Boolean) {
        if (prepared.providerRefundId == null &&
            OffsetDateTime.now().isAfter(prepared.requestedAt.plus(PROVIDER_RETRY_WINDOW))
        ) {
            transactionTemplate.executeWithoutResult {
                val dispute = disputeRepository.findByIdForUpdate(prepared.disputeId)
                    ?: return@executeWithoutResult
                if (dispute.providerRefundId == null && dispute.refundStatus in IN_FLIGHT_STATUSES) {
                    dispute.refundStatus = MANUAL_REVIEW
                    dispute.status = DisputeStatus.ESCALATED
                    disputeRepository.save(dispute)
                }
            }
            log.error("Refund {} has an unknown provider outcome and requires manual reconciliation", prepared.disputeId)
            if (reportFailure) {
                throw checkoutConflict("Refund outcome is unknown. Reconcile it in PayMongo before retrying.")
            }
            return
        }
        try {
            val paymentId = prepared.paymentId ?: recoverPaymentId(prepared)
            val response = prepared.providerRefundId?.let(paymongoClient::retrieveRefund)
                ?: paymongoClient.createRefund(
                    request = PaymongoRefundRequest(
                        PaymongoRefundRequestEnvelope(
                            PaymongoRefundRequestAttributes(
                                amount = prepared.amount.movePointRight(2).longValueExact(),
                                paymentId = paymentId,
                                notes = "QuickPitik dispute ${prepared.disputeId}".take(255),
                            ),
                        ),
                    ),
                    idempotencyKey = "${prepared.disputeId}:${prepared.requestedAt.toInstant().toEpochMilli()}",
                )
            val status = response.data.attributes.status.lowercase()
            transactionTemplate.executeWithoutResult {
                val dispute = disputeRepository.findByIdForUpdate(prepared.disputeId) ?: return@executeWithoutResult
                dispute.providerRefundId = response.data.id.ifBlank { dispute.providerRefundId }
                dispute.refundStatus = status.ifBlank { REQUESTING }
                disputeRepository.save(dispute)
                if (status == SUCCEEDED) completeSucceededRefund(dispute)
            }
            if (status == FAILED && reportFailure) {
                throw checkoutConflict("PayMongo could not process this refund.")
            }
        } catch (ex: ConflictException) {
            throw ex
        } catch (ex: Exception) {
            log.warn("PayMongo refund {} deferred: {}", prepared.disputeId, ex.message)
            if (reportFailure) {
                throw ConflictException(
                    message = "Payment gateway could not start the refund - try again in a moment.",
                    code = ErrorCodes.PAYMENT_FAILED,
                )
            }
        }
    }

    private fun recoverPaymentId(prepared: PreparedRefund): String {
        val checkout = paymongoClient.retrieveCheckoutSession(prepared.checkoutSessionId)
        val paymentId = checkout.data.attributes.payments
            .firstOrNull { it.attributes.status.equals("paid", ignoreCase = true) }
            ?.id
            ?.takeIf { it.isNotBlank() }
            ?: throw checkoutConflict("PayMongo Checkout Session has no paid payment resource.")
        transactionTemplate.executeWithoutResult {
            paymentRepository.findAllByProviderAndProviderRefForUpdate(PAYMONGO, prepared.checkoutSessionId)
                .forEach { payment ->
                    payment.providerPaymentId = paymentId
                    paymentRepository.save(payment)
                }
        }
        return paymentId
    }

    private fun completeSucceededRefund(dispute: Dispute) {
        if (dispute.status == DisputeStatus.RESOLVED && dispute.refundedAt != null) return
        val resolution = dispute.resolution ?: return
        val refundAmount = dispute.refundAmountPhp ?: return
        val now = OffsetDateTime.now()

        dispute.refundStatus = SUCCEEDED
        dispute.refundedAt = now
        dispute.status = DisputeStatus.RESOLVED
        dispute.resolvedAt = now
        disputeRepository.save(dispute)

        val photo = photoRepository.findById(dispute.photoId).orElse(null)
        photo?.photographerId?.let { mintRefundTransaction(dispute, it, refundAmount) }

        if (resolution == DisputeResolution.REFUND_FULL) {
            downloadGrantRepository.deleteById(DownloadGrantId(dispute.orderId, dispute.photoId))
        }
        updateOrderRefundStatus(dispute.orderId)
        publishOutcome(dispute, resolution, refundAmount, photo?.photographerId)
    }

    private fun updateOrderRefundStatus(orderId: UUID) {
        val order = orderRepository.findByIdForUpdate(orderId) ?: return
        val photoIds = orderItemRepository.findByIdOrderId(orderId).map { it.id.photoId }.toSet()
        val fullyRefunded = disputeRepository.findByOrderId(orderId)
            .filter { it.refundStatus == SUCCEEDED && it.resolution == DisputeResolution.REFUND_FULL }
            .map { it.photoId }
            .toSet()
        order.status = if (photoIds.isNotEmpty() && fullyRefunded.containsAll(photoIds)) {
            OrderStatus.REFUNDED
        } else {
            OrderStatus.FULFILLED
        }
        orderRepository.save(order)
    }

    private fun publishOutcome(
        dispute: Dispute,
        resolution: DisputeResolution,
        refundAmount: BigDecimal,
        photographerId: UUID?,
    ) {
        val adminId = dispute.refundRequestedBy ?: return
        val decision = adminDecisionLogService.logDisputeDecision(
            adminId = adminId,
            targetDisputeId = dispute.id,
            decision = "resolved",
            reason = dispute.refundReason,
            meta = mapOf(
                "resolution" to resolution.wire,
                "refundAmount" to refundAmount.toPlainString(),
                "providerRefundId" to dispute.providerRefundId,
            ),
        )
        photographerId?.let {
            adminDecisionLogService.pushMessage(
                photographerId = it,
                kind = PhotographerMessageKind.DISPUTE_RESOLVED,
                body = "A dispute on photo ${dispute.photoId} was resolved as ${resolution.wire}.",
                sourceAdminId = adminId,
                sourceDecisionId = decision.id,
            )
        }
        dispute.runnerId?.let { runnerId ->
            runnerMessagesService.pushMessage(
                runnerId = runnerId,
                kind = RunnerMessageKind.DISPUTE_RESOLVED,
                body = "Your refund has been approved - PHP ${refundAmount.toPlainString()} refunded.",
                sourceAdminId = adminId,
                sourceDecisionId = decision.id,
                orderId = dispute.orderId,
            )
        }
    }

    private fun mintRefundTransaction(dispute: Dispute, photographerId: UUID, amount: BigDecimal) {
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

    private fun validatePartialAmount(requested: BigDecimal?, fullPrice: BigDecimal): BigDecimal {
        val amount = requested ?: throw ValidationException(
            message = "refundAmount is required for partial refunds",
            code = ErrorCodes.REFUND_AMOUNT_REQUIRED,
            field = "refundAmount",
        )
        if (amount <= BigDecimal.ZERO || amount >= fullPrice) {
            throw ValidationException(
                message = "refundAmount must be greater than zero and less than the item price",
                code = ErrorCodes.VALIDATION_ERROR,
                field = "refundAmount",
            )
        }
        return runCatching { amount.setScale(2, RoundingMode.UNNECESSARY) }.getOrElse {
            throw ValidationException(
                message = "refundAmount must have at most two decimal places",
                code = ErrorCodes.VALIDATION_ERROR,
                field = "refundAmount",
            )
        }
    }

    private fun checkoutConflict(message: String): ConflictException =
        ConflictException(message = message, code = ErrorCodes.CONFLICT)

    private data class PreparedRefund(
        val disputeId: UUID,
        val paymentId: String?,
        val checkoutSessionId: String,
        val amount: BigDecimal,
        val requestedAt: OffsetDateTime,
        val providerRefundId: String?,
        val completed: Boolean = false,
    )

    private companion object {
        const val PAYMONGO = "paymongo"
        const val REQUESTING = "requesting"
        const val SUCCEEDED = "succeeded"
        const val FAILED = "failed"
        const val MANUAL_REVIEW = "manual_review"
        val IN_FLIGHT_STATUSES = listOf(REQUESTING, "pending", "processing")
        val PROVIDER_RETRY_WINDOW: Duration = Duration.ofHours(23)
        const val BATCH_SIZE = 100
    }
}
