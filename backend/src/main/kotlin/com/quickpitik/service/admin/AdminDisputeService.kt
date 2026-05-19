package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.AdminDisputeDto
import com.quickpitik.dto.admin.DisputeActivityEntry
import com.quickpitik.dto.admin.DisputeOrderSnapshotDto
import com.quickpitik.dto.admin.DisputePhotoSnapshotDto
import com.quickpitik.dto.admin.ResolveDisputeRequest
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeResolution
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.RunnerMessageKind
import com.quickpitik.entity.Transaction
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.service.storage.StorageService
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.math.RoundingMode
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional
class AdminDisputeService(
    private val disputeRepository: DisputeRepository,
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val photoRepository: PhotoRepository,
    private val eventRepository: EventRepository,
    private val userRepository: UserRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val transactionRepository: TransactionRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val platformProperties: PlatformProperties,
    private val adminDecisionLogService: AdminDecisionLogService,
    private val adminDecisionLogRepository: AdminDecisionLogRepository,
    private val runnerMessagesService: RunnerMessagesService,
) {

    @Transactional(readOnly = true)
    fun list(
        statusFilter: String?,
        params: PaginationParams,
    ): PaginatedResponse<AdminDisputeDto> {
        val statusWire = statusFilter?.takeIf { it.isNotBlank() }?.let { parseStatus(it).wire }
        val page = disputeRepository.pageForAdmin(statusWire, OffsetLimitPageable(params))
        if (page.isEmpty) return PaginatedResponse.empty(params)
        val items = hydrateMany(page.content)
        return PaginatedResponse.of(items, page.totalElements, params)
    }

    fun resolve(adminId: UUID, disputeId: UUID, req: ResolveDisputeRequest): AdminDisputeDto {
        val resolution = parseResolution(req.resolution)
        val dispute = disputeRepository.findById(disputeId).orElseThrow {
            NotFoundException(code = ErrorCodes.DISPUTE_NOT_FOUND, message = "Dispute not found")
        }
        if (dispute.status == DisputeStatus.RESOLVED || dispute.status == DisputeStatus.DENIED) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Dispute is already ${dispute.status.wire}",
            )
        }

        // Refund-bearing resolutions need an amount or fall back to full
        // refund of the order_item price.
        val refundAmount = computeRefundAmount(dispute, resolution, req.refundAmount)

        dispute.resolution = resolution
        dispute.status = if (resolution == DisputeResolution.DENY) DisputeStatus.DENIED else DisputeStatus.RESOLVED
        dispute.refundAmountPhp = refundAmount
        dispute.resolvedAt = OffsetDateTime.now()
        disputeRepository.save(dispute)

        // Materialize the refund as a negative-amount transaction row per
        // Q-E3 so SUM(amount_kept_php) is net-of-refunds without a parallel
        // ledger. Skipped on deny.
        val photographerId = dispute.runnerId?.let { /* photographerId lives on Photo, not dispute */ null }
            ?: photoRepository.findById(dispute.photoId).map { it.photographerId }.orElse(null)

        if (photographerId != null && resolution != DisputeResolution.DENY && refundAmount != null) {
            mintRefundTx(dispute, photographerId, refundAmount)
        }

        val decision = adminDecisionLogService.logDisputeDecision(
            adminId = adminId,
            targetDisputeId = dispute.id,
            decision = "resolved",
            reason = req.reason,
            meta = buildMap {
                put("resolution", resolution.wire)
                refundAmount?.let { put("refundAmount", it.toPlainString()) }
            },
        )
        if (photographerId != null) {
            adminDecisionLogService.pushMessage(
                photographerId = photographerId,
                kind = PhotographerMessageKind.DISPUTE_RESOLVED,
                body = "A dispute on photo ${dispute.photoId} was resolved as ${resolution.wire}.",
                sourceAdminId = adminId,
                sourceDecisionId = decision.id,
            )
        }
        pushRunnerOutcome(
            dispute = dispute,
            kind = RunnerMessageKind.DISPUTE_RESOLVED,
            body = buildResolvedBody(resolution, refundAmount),
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateOne(dispute)
    }

    fun deny(adminId: UUID, disputeId: UUID, reason: String?): AdminDisputeDto {
        val dispute = disputeRepository.findById(disputeId).orElseThrow {
            NotFoundException(code = ErrorCodes.DISPUTE_NOT_FOUND, message = "Dispute not found")
        }
        if (dispute.status == DisputeStatus.RESOLVED || dispute.status == DisputeStatus.DENIED) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Dispute is already ${dispute.status.wire}",
            )
        }
        dispute.status = DisputeStatus.DENIED
        dispute.resolution = DisputeResolution.DENY
        dispute.refundAmountPhp = null
        dispute.resolvedAt = OffsetDateTime.now()
        disputeRepository.save(dispute)

        val photographerId = photoRepository.findById(dispute.photoId).map { it.photographerId }.orElse(null)
        val decision = adminDecisionLogService.logDisputeDecision(
            adminId = adminId,
            targetDisputeId = dispute.id,
            decision = "denied",
            reason = reason,
        )
        if (photographerId != null) {
            adminDecisionLogService.pushMessage(
                photographerId = photographerId,
                kind = PhotographerMessageKind.DISPUTE_DENIED,
                body = "A dispute on photo ${dispute.photoId} was denied.",
                sourceAdminId = adminId,
                sourceDecisionId = decision.id,
            )
        }
        pushRunnerOutcome(
            dispute = dispute,
            kind = RunnerMessageKind.DISPUTE_DENIED,
            body = "Your refund request for photo ${shortPhotoId(dispute.photoId)} was declined." +
                (reason?.takeIf { it.isNotBlank() }?.let { " Reason: $it" } ?: ""),
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateOne(dispute)
    }

    fun escalate(adminId: UUID, disputeId: UUID, reason: String?): AdminDisputeDto {
        val dispute = disputeRepository.findById(disputeId).orElseThrow {
            NotFoundException(code = ErrorCodes.DISPUTE_NOT_FOUND, message = "Dispute not found")
        }
        if (dispute.status == DisputeStatus.RESOLVED || dispute.status == DisputeStatus.DENIED) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Cannot escalate a ${dispute.status.wire} dispute",
            )
        }
        dispute.status = DisputeStatus.ESCALATED
        disputeRepository.save(dispute)

        val photographerId = photoRepository.findById(dispute.photoId).map { it.photographerId }.orElse(null)
        val decision = adminDecisionLogService.logDisputeDecision(
            adminId = adminId,
            targetDisputeId = dispute.id,
            decision = "escalated",
            reason = reason,
        )
        if (photographerId != null) {
            adminDecisionLogService.pushMessage(
                photographerId = photographerId,
                kind = PhotographerMessageKind.DISPUTE_ESCALATED,
                body = "A dispute on photo ${dispute.photoId} was escalated for further review.",
                sourceAdminId = adminId,
                sourceDecisionId = decision.id,
            )
        }
        pushRunnerOutcome(
            dispute = dispute,
            kind = RunnerMessageKind.DISPUTE_ESCALATED,
            body = "Your refund request for photo ${shortPhotoId(dispute.photoId)} is being escalated for further review.",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateOne(dispute)
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    /**
     * Push the dispute outcome to the runner's inbox. Skipped silently
     * when `dispute.runnerId` is null — that's a guest order, no runner
     * account to notify (the runner gets the email receipt at
     * /orders/return instead).
     */
    private fun pushRunnerOutcome(
        dispute: Dispute,
        kind: RunnerMessageKind,
        body: String,
        sourceAdminId: UUID,
        sourceDecisionId: UUID,
    ) {
        val runnerId = dispute.runnerId ?: return
        runnerMessagesService.pushMessage(
            runnerId = runnerId,
            kind = kind,
            body = body,
            sourceAdminId = sourceAdminId,
            sourceDecisionId = sourceDecisionId,
            orderId = dispute.orderId,
        )
    }

    private fun buildResolvedBody(
        resolution: DisputeResolution,
        refundAmount: BigDecimal?,
    ): String = when (resolution) {
        DisputeResolution.REFUND_FULL,
        DisputeResolution.REFUND_PARTIAL,
        -> {
            val amount = refundAmount?.toPlainString()?.let { "₱$it refunded" } ?: "Refund processing"
            "Your refund has been approved — $amount."
        }
        DisputeResolution.DENY -> "Your refund request was declined."
    }

    private fun shortPhotoId(photoId: UUID): String =
        photoId.toString().take(8).uppercase()

    // ─── Existing helpers ─────────────────────────────────────────────────
    private fun computeRefundAmount(
        dispute: Dispute,
        resolution: DisputeResolution,
        fromRequest: BigDecimal?,
    ): BigDecimal? {
        if (resolution == DisputeResolution.DENY) return null
        // FULL refund = the photo's per-item paid price.
        val itemPrice = orderItemRepository.findByIdOrderId(dispute.orderId)
            .firstOrNull { it.id.photoId == dispute.photoId }
            ?.pricePhpAtPurchase
        return when (resolution) {
            DisputeResolution.REFUND_FULL -> itemPrice ?: fromRequest
            DisputeResolution.REFUND_PARTIAL -> fromRequest
                ?: throw ValidationException(
                    code = ErrorCodes.REFUND_AMOUNT_REQUIRED,
                    message = "refundAmount is required for partial refunds",
                    field = "refundAmount",
                )
            DisputeResolution.DENY -> null
        }
    }

    private fun mintRefundTx(dispute: Dispute, photographerId: UUID, refundAmount: BigDecimal) {
        val originating = transactionRepository.findByOrderIdAndPhotoIdAndIsRefund(
            orderId = dispute.orderId,
            photoId = dispute.photoId,
            isRefund = false,
        )
        val existing = transactionRepository.findByOrderIdAndPhotoIdAndIsRefund(
            orderId = dispute.orderId,
            photoId = dispute.photoId,
            isRefund = true,
        )
        if (existing != null) return

        val keepRate = platformProperties.photographerKeepRate
        val keptPortionRefunded = refundAmount
            .multiply(keepRate)
            .setScale(2, RoundingMode.HALF_UP)
            .negate()

        val order = orderRepository.findById(dispute.orderId).orElse(null) ?: return
        try {
            transactionRepository.save(
                Transaction(
                    paidAt = order.paidAt ?: order.createdAt,
                    photographerId = photographerId,
                    eventId = order.eventId,
                    photoId = dispute.photoId,
                    orderId = dispute.orderId,
                    buyerId = order.userId,
                    buyerDisplayName = originating?.buyerDisplayName ?: "",
                    amountKeptPhp = keptPortionRefunded,
                    isRefund = true,
                    refundOf = originating?.id,
                ),
            )
        } catch (ex: DataIntegrityViolationException) {
            // Concurrent refund mint — clean no-op, the unique
            // (order_id, photo_id, is_refund=true) index caught it.
        }
    }

    private fun hydrateMany(disputes: List<Dispute>): List<AdminDisputeDto> {
        if (disputes.isEmpty()) return emptyList()
        // Batch-fetch activity for the page in one round trip; pass the
        // pre-grouped map to hydrateOne so the per-row path doesn't re-query.
        val activityByDispute = loadActivityForDisputes(disputes.map { it.id })
        return disputes.map { hydrateOne(it, activityByDispute[it.id].orEmpty()) }
    }

    private fun loadActivityForDisputes(
        disputeIds: Collection<UUID>,
    ): Map<UUID, List<DisputeActivityEntry>> {
        if (disputeIds.isEmpty()) return emptyMap()
        return adminDecisionLogRepository
            .findByTargetDisputeIdInOrderByDecidedAtDesc(disputeIds)
            .mapNotNull { row -> row.targetDisputeId?.let { it to toActivityEntry(row) } }
            .groupBy({ it.first }, { it.second })
    }

    private fun toActivityEntry(row: AdminDecisionLog): DisputeActivityEntry {
        // meta JSONB carries resolution-specific fields written by
        // AdminDisputeService.resolve(). Flatten the two we surface; keep
        // every other meta key inside the JSONB blob for future use.
        val meta = row.meta
        val resolution = (meta?.get("resolution") as? String)?.takeIf { it.isNotBlank() }
        val refundAmount = (meta?.get("refundAmount") as? String)?.let {
            runCatching { BigDecimal(it) }.getOrNull()
        } ?: (meta?.get("refundAmount") as? Number)?.let { BigDecimal(it.toString()) }
        return DisputeActivityEntry(
            id = row.id,
            decidedAt = row.decidedAt,
            decision = row.decision,
            resolution = resolution,
            refundAmount = refundAmount,
            reason = row.reason,
        )
    }

    internal fun hydrateOne(
        dispute: Dispute,
        activity: List<DisputeActivityEntry> = loadActivityForDisputes(listOf(dispute.id))[dispute.id].orEmpty(),
    ): AdminDisputeDto {
        val order = orderRepository.findById(dispute.orderId).orElse(null)
        val photo = photoRepository.findById(dispute.photoId).orElse(null)

        val runnerHandle = dispute.runnerId?.let { rid ->
            userRepository.findById(rid).map { runnerDisplayHandle(it.name, it.email) }.orElse("")
        } ?: order?.let { runnerDisplayHandle("", it.recipientEmail) }
        ?: ""

        val photographerId = photo?.photographerId
        val photographerHandle: String = photographerId?.let {
            photographerSettingsRepository.findById(it).map { s -> s.handle }.orElse(null)
        }.orEmpty()

        val eventId = order?.eventId ?: photo?.eventId ?: dispute.orderId // fallback for ghost rows
        val eventName = (order?.eventId ?: photo?.eventId)?.let { eid ->
            eventRepository.findById(eid).map { it.name }.orElse(null)
        }

        return AdminDisputeDto(
            id = dispute.id,
            orderId = dispute.orderId,
            photoId = dispute.photoId,
            eventId = eventId,
            eventName = eventName,
            runnerHandle = runnerHandle,
            photographerHandle = photographerHandle,
            reason = dispute.reasonWire,
            note = dispute.note,
            status = dispute.status.wire,
            reportedAt = dispute.openedAt.toString(),
            resolvedAt = dispute.resolvedAt?.toString(),
            refundAmount = dispute.refundAmountPhp,
            resolution = dispute.resolutionWire,
            orderSnapshot = DisputeOrderSnapshotDto(
                total = order?.totalPhp ?: BigDecimal.ZERO,
                paymentMethod = order?.paymentMethodWire ?: "",
                paidAt = order?.paidAt?.toString(),
            ),
            photoSnapshot = DisputePhotoSnapshotDto(
                alt = photo?.altText ?: "",
                kmMark = photo?.km,
                bib = photo?.bibs?.minByOrNull { it.bibNumber }?.bibNumber,
                thumbnailUrl = photo?.let {
                    val key = it.thumbnailS3Key ?: it.watermarkS3Key ?: it.s3Key
                    storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
                },
            ),
            activity = activity,
        )
    }

    private fun parseStatus(raw: String): DisputeStatus =
        runCatching { DisputeStatus.fromWire(raw.trim().lowercase()) }.getOrElse {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "status must be one of open / resolved / denied / escalated",
                field = "status",
            )
        }

    private fun parseResolution(raw: String): DisputeResolution =
        runCatching { DisputeResolution.fromWire(raw.trim().lowercase()) }.getOrElse {
            throw ValidationException(
                code = ErrorCodes.INVALID_RESOLUTION,
                message = "resolution must be one of refund_full / refund_partial / deny",
                field = "resolution",
            )
        }
}
