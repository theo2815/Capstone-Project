package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PaymongoProperties
import com.quickpitik.common.toCentavos
import com.quickpitik.dto.orders.PaymongoRefundRequest
import com.quickpitik.dto.orders.PaymongoRefundRequestAttributes
import com.quickpitik.dto.orders.PaymongoRefundRequestEnvelope
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.REFUND_LINK_SENT
import com.quickpitik.entity.REFUND_SENT_TO_MANUAL_REVIEW
import com.quickpitik.entity.DisputeResolution
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.DownloadGrantId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Payment
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.RunnerMessageKind
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.earnings.TransactionMintingService
import com.quickpitik.service.runner.RunnerMessagesService
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import org.springframework.web.client.RestClientResponseException
import java.math.BigDecimal
import java.math.RoundingMode
import java.net.URI
import java.net.URISyntaxException
import java.time.Duration
import java.time.OffsetDateTime
import java.time.temporal.ChronoUnit
import java.util.UUID

@Service
class PaymongoRefundService(
    private val disputeRepository: DisputeRepository,
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val paymentRepository: PaymentRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val photoRepository: PhotoRepository,
    private val transactionMintingService: TransactionMintingService,
    private val paymongoClient: PaymongoClient,
    private val adminDecisionLogService: AdminDecisionLogService,
    private val runnerMessagesService: RunnerMessagesService,
    private val transactionTemplate: TransactionTemplate,
    private val paymongoProperties: PaymongoProperties,
    private val events: ApplicationEventPublisher,
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
        if (!prepared.completed && !prepared.parked) process(prepared, reportFailure = true)
    }

    // The admin refunded a parked QR Ph payment in the PayMongo dashboard and pastes the refund transfer link here.
    // The link is mailed to the runner, so it must start with a configured PayMongo prefix (anti-phishing); that check
    // runs before the lock. Calling again replaces the link (expired-link case) and tells the runner again.
    fun sendRefundLink(adminId: UUID, disputeId: UUID, url: String) {
        val link = url.trim()
        if (link.isEmpty() || link.length > MAX_REFUND_LINK_LENGTH || link.any { it.isWhitespace() } ||
            // A prefix must end in "/" or "https://host" would also match "https://host.evil.example".
            paymongoProperties.refundLinkAllowedPrefixes.none { it.endsWith("/") && link.startsWith(it) } ||
            !isPlainHttpsUrl(link)
        ) {
            throw ValidationException(
                message = "Paste the refund transfer link from the PayMongo dashboard.",
                code = ErrorCodes.VALIDATION_ERROR,
                field = "url",
            )
        }
        transactionTemplate.executeWithoutResult {
            val dispute = disputeRepository.findByIdForUpdate(disputeId)
                ?: throw NotFoundException("Dispute not found", ErrorCodes.DISPUTE_NOT_FOUND)
            if (dispute.refundStatus != MANUAL_REVIEW) {
                throw checkoutConflict("This dispute is not waiting for a manual refund.")
            }
            val amount = dispute.refundAmountPhp ?: throw checkoutConflict("This dispute has no approved refund amount.")
            dispute.refundLinkUrl = link
            dispute.refundLinkSentAt = OffsetDateTime.now().truncatedTo(ChronoUnit.MICROS)
            disputeRepository.save(dispute)
            // The url in meta is the audit trail: this link goes to a customer.
            val decision = adminDecisionLogService.logDisputeDecision(
                adminId = adminId,
                targetDisputeId = dispute.id,
                decision = REFUND_LINK_SENT,
                reason = null,
                meta = mapOf("refundLinkUrl" to link),
            )
            dispute.runnerId?.let { runnerId ->
                runnerMessagesService.pushMessage(
                    runnerId = runnerId,
                    kind = RunnerMessageKind.ADMIN_MESSAGE,
                    title = "Your refund link is ready",
                    body = "Claim your refund of PHP ${amount.toPlainString()} with this PayMongo link: $link " +
                        "The link works for 3 days.",
                    sourceAdminId = adminId,
                    sourceDecisionId = decision.id,
                    orderId = dispute.orderId,
                )
            }
            events.publishEvent(RefundLinkSentEvent(dispute.orderId, link, amount))
        }
    }

    fun handleWebhook(providerRefundId: String, status: String): Boolean =
        transactionTemplate.execute {
            val dispute = disputeRepository.findByProviderRefundId(providerRefundId) ?: return@execute false
            val locked = disputeRepository.findByIdForUpdate(dispute.id) ?: return@execute false
            if (locked.providerRefundId != providerRefundId) return@execute false
            if (isCompletedRefund(locked)) return@execute true
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
        if (isCompletedRefund(dispute)) {
            if (dispute.refundStatus != SUCCEEDED) {
                dispute.refundStatus = SUCCEEDED
                disputeRepository.save(dispute)
            }
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
        val payment = paymentRepository.findByOrderId(dispute.orderId).settledPaymongo()
            ?: throw checkoutConflict("The order has no settled PayMongo payment to refund.")
        val providerRef = payment.providerRef
            ?: throw checkoutConflict("The settled payment has no PayMongo reference.")

        // TIMESTAMPTZ keeps microseconds; store what the database will give back (INC-12).
        val now = OffsetDateTime.now().truncatedTo(ChronoUnit.MICROS)
        // QR Ph payments cannot be refunded through the API. PayMongo's dashboard refunds a whole payment
        // only, and one payment covers the whole checkout, so approval needs the refund to equal that total.
        val qrph = orderRepository.findById(dispute.orderId).orElse(null)?.paymentMethod == PaymentMethod.QRPH
        if (qrph) {
            val checkoutTotal = paymentRepository.findAllByProviderAndProviderRef(PAYMONGO, providerRef)
                .fold(BigDecimal.ZERO) { sum, it -> sum + it.amountPhp }
            if (amount.compareTo(checkoutTotal) != 0) {
                throw checkoutConflict(
                    "PayMongo's dashboard can only refund the whole payment. Deny this request with a reason instead.",
                )
            }
        }
        dispute.resolution = resolution
        dispute.refundAmountPhp = amount
        dispute.refundStatus = if (qrph) MANUAL_REVIEW else REQUESTING
        dispute.refundRequestedAt = now
        dispute.refundRequestedBy = adminId
        dispute.refundReason = reason?.trim()?.takeIf { it.isNotEmpty() }
        dispute.providerRefundId = null
        if (qrph) dispute.status = DisputeStatus.ESCALATED
        disputeRepository.save(dispute)
        if (qrph) {
            val decision = logManualReviewHandoff(adminId, dispute)
            dispute.runnerId?.let { runnerId ->
                runnerMessagesService.pushMessage(
                    runnerId = runnerId,
                    kind = RunnerMessageKind.ADMIN_MESSAGE,
                    title = "Refund approved",
                    body = "Your refund of PHP ${amount.toPlainString()} was approved. " +
                        "We will send you a PayMongo link to claim it.",
                    sourceAdminId = adminId,
                    sourceDecisionId = decision.id,
                    orderId = dispute.orderId,
                )
            }
        }
        return PreparedRefund(
            disputeId = dispute.id,
            paymentId = payment.providerPaymentId,
            providerRef = providerRef,
            amount = amount,
            requestedAt = now,
            providerRefundId = null,
            parked = qrph,
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
            providerRef = payment.providerRef
                ?: throw checkoutConflict("The payment has no PayMongo reference."),
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
            markManualReview(prepared)
            log.error("Refund {} has an unknown provider outcome and requires manual reconciliation", prepared.disputeId)
            if (reportFailure) {
                throw checkoutConflict("Refund outcome is unknown. Reconcile it in PayMongo before retrying.")
            }
            return
        }
        try {
            val paymentId = if (prepared.providerRef.startsWith("pi_")) {
                recoverPaymentId(prepared)
            } else {
                prepared.paymentId ?: recoverPaymentId(prepared)
            }
            val response = prepared.providerRefundId?.let(paymongoClient::retrieveRefund)
                ?: paymongoClient.createRefund(
                    request = PaymongoRefundRequest(
                        PaymongoRefundRequestEnvelope(
                            PaymongoRefundRequestAttributes(
                                amount = prepared.amount.toCentavos(),
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
                if (isCompletedRefund(dispute) || dispute.refundRequestedAt?.isEqual(prepared.requestedAt) != true) {
                    return@executeWithoutResult
                }
                if (dispute.providerRefundId != null &&
                    response.data.id.isNotBlank() &&
                    dispute.providerRefundId != response.data.id
                ) {
                    return@executeWithoutResult
                }
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
            if (ex is RestClientResponseException && ex.statusCode.value() == 400) {
                markManualReview(prepared)
                log.error("PayMongo rejected refund {}: {}", prepared.disputeId, ex.responseBodyAsString)
                if (reportFailure) {
                    throw ConflictException(
                        message = "PayMongo rejected this refund. Review the payment manually.",
                        code = ErrorCodes.PAYMENT_FAILED,
                    )
                }
                return
            }
            log.warn("PayMongo refund {} deferred: {}", prepared.disputeId, ex.message)
            if (reportFailure) {
                throw ConflictException(
                    message = "Payment gateway could not start the refund - try again in a moment.",
                    code = ErrorCodes.PAYMENT_FAILED,
                )
            }
        }
    }

    private fun markManualReview(prepared: PreparedRefund) {
        transactionTemplate.executeWithoutResult {
            val dispute = disputeRepository.findByIdForUpdate(prepared.disputeId) ?: return@executeWithoutResult
            if (!isCompletedRefund(dispute) &&
                dispute.refundRequestedAt?.isEqual(prepared.requestedAt) == true &&
                dispute.providerRefundId == prepared.providerRefundId &&
                dispute.refundStatus in IN_FLIGHT_STATUSES
            ) {
                dispute.refundStatus = MANUAL_REVIEW
                dispute.status = DisputeStatus.ESCALATED
                disputeRepository.save(dispute)
                dispute.refundRequestedBy?.let { logManualReviewHandoff(it, dispute) }
            }
        }
    }

    // Reason stays null: no money has moved yet, and the admin's dispute history shows a reason as a note.
    // The meta key is not "resolution" (the admin history reads that as a final outcome).
    private fun logManualReviewHandoff(adminId: UUID, dispute: Dispute) =
        adminDecisionLogService.logDisputeDecision(
            adminId = adminId,
            targetDisputeId = dispute.id,
            decision = REFUND_SENT_TO_MANUAL_REVIEW,
            reason = null,
            meta = mapOf(
                "requestedResolution" to dispute.resolution?.wire,
                "refundAmount" to dispute.refundAmountPhp?.toPlainString(),
            ),
        )

    private fun recoverPaymentId(prepared: PreparedRefund): String {
        val paymentId = if (prepared.providerRef.startsWith("pi_")) {
            val intent = paymongoClient.retrievePaymentIntent(prepared.providerRef)
            intent.data.attributes.payments.takeIf { intent.data.attributes.status == "succeeded" }
                ?.paidPaymentId()
        } else {
            paymongoClient.retrieveCheckoutSession(prepared.providerRef).data.attributes.payments.paidPaymentId()
        }
        val paidPaymentId = paymentId
            ?.takeIf { it.isNotBlank() }
            ?: throw checkoutConflict("PayMongo payment has no paid payment resource.")
        transactionTemplate.executeWithoutResult {
            paymentRepository.findAllByProviderAndProviderRefForUpdate(PAYMONGO, prepared.providerRef)
                .forEach { payment ->
                    payment.providerPaymentId = paidPaymentId
                    paymentRepository.save(payment)
                }
        }
        return paidPaymentId
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
        photo?.photographerId?.let { transactionMintingService.mintRefund(dispute, refundAmount, it) }

        if (resolution == DisputeResolution.REFUND_FULL) {
            downloadGrantRepository.deleteById(DownloadGrantId(dispute.orderId, dispute.photoId))
        }
        updateOrderSettlementStatus(dispute.orderId)
        publishOutcome(dispute, resolution, refundAmount, photo?.photographerId)
    }

    private fun updateOrderSettlementStatus(orderId: UUID) {
        val order = orderRepository.findByIdForUpdate(orderId) ?: return
        val photoIds = orderItemRepository.findByIdOrderId(orderId).map { it.id.photoId }.toSet()
        val fullyRefunded = disputeRepository.findByOrderId(orderId)
            .filter { it.refundStatus == SUCCEEDED && it.resolution == DisputeResolution.REFUND_FULL }
            .map { it.photoId }
            .toSet()
        order.status = if (photoIds.isNotEmpty() && fullyRefunded.containsAll(photoIds)) {
            OrderStatus.REFUNDED
        } else if (order.status == OrderStatus.REFUNDED) {
            OrderStatus.FULFILLED
        } else {
            order.status
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

    // URI() rejects quotes, angle brackets, backslashes and control characters, which would break out of the e-mail href.
    private fun isPlainHttpsUrl(link: String): Boolean = try {
        URI(link).let { it.scheme == "https" && it.userInfo == null }
    } catch (_: URISyntaxException) {
        false
    }

    private fun checkoutConflict(message: String): ConflictException =
        ConflictException(message = message, code = ErrorCodes.CONFLICT)

    private fun isCompletedRefund(dispute: Dispute): Boolean =
        dispute.status == DisputeStatus.RESOLVED && dispute.refundedAt != null

    private data class PreparedRefund(
        val disputeId: UUID,
        val paymentId: String?,
        val providerRef: String,
        val amount: BigDecimal,
        val requestedAt: OffsetDateTime,
        val providerRefundId: String?,
        val completed: Boolean = false,
        // Parked for a manual refund in PayMongo's dashboard: nothing to send to the API.
        val parked: Boolean = false,
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
        const val MAX_REFUND_LINK_LENGTH = 500
    }
}

// The order's settled PayMongo payment: the one a refund targets.
internal fun Iterable<Payment>.settledPaymongo(): Payment? =
    firstOrNull { it.provider == "paymongo" && it.status == PaymentStatus.SUCCEEDED }
