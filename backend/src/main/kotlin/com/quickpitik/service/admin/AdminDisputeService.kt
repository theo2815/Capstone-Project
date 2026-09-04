package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
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
import com.quickpitik.entity.Event
import com.quickpitik.entity.Order
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.RunnerMessageKind
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.service.orders.PaymongoRefundService
import com.quickpitik.service.storage.StorageService
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional
class AdminDisputeService(
    private val disputeRepository: DisputeRepository,
    private val orderRepository: OrderRepository,
    private val photoRepository: PhotoRepository,
    private val eventRepository: EventRepository,
    private val userRepository: UserRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val adminDecisionLogService: AdminDecisionLogService,
    private val adminDecisionLogRepository: AdminDecisionLogRepository,
    private val runnerMessagesService: RunnerMessagesService,
    private val paymongoRefundService: PaymongoRefundService,
    private val transactionTemplate: TransactionTemplate,
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

    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    fun resolve(adminId: UUID, disputeId: UUID, req: ResolveDisputeRequest): AdminDisputeDto {
        val resolution = parseResolution(req.resolution)
        if (resolution == DisputeResolution.DENY) {
            return transactionTemplate.execute { deny(adminId, disputeId, req.reason) }
                ?: error("Deny resolution returned null")
        }
        paymongoRefundService.request(adminId, disputeId, resolution, req.refundAmount, req.reason)
        return transactionTemplate.execute {
            val dispute = disputeRepository.findById(disputeId).orElseThrow {
                NotFoundException(code = ErrorCodes.DISPUTE_NOT_FOUND, message = "Dispute not found")
            }
            hydrateOne(dispute)
        } ?: error("Refund resolution returned null")
    }

    /**
     * `reason` is **runner-facing**, deliberately. It is interpolated into the
     * inbox message below, and every dispute decision's reason also reaches
     * the runner as `RunnerDisputeDto.resolutionNote` on /me/orders (see
     * `OrderService.hydrateDisputesByOrderId`), which the website renders in
     * the refund timeline. Admins should write it as an explanation to the
     * customer, not as an internal note.
     *
     * A 2026-05-27 audit filed this as a leak and proposed dropping the
     * interpolation. That would have left the identical text visible on
     * /orders, so the two runner surfaces would have disagreed for no gain.
     * Confirmed intended 2026-08-14 — do not "fix" it back.
     */
    fun deny(adminId: UUID, disputeId: UUID, reason: String?): AdminDisputeDto {
        val dispute = disputeRepository.findByIdForUpdate(disputeId) ?: throw
            NotFoundException(code = ErrorCodes.DISPUTE_NOT_FOUND, message = "Dispute not found")
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

    private fun shortPhotoId(photoId: UUID): String =
        photoId.toString().take(8).uppercase()

    // ─── Existing helpers ─────────────────────────────────────────────────
    private fun hydrateMany(disputes: List<Dispute>): List<AdminDisputeDto> {
        if (disputes.isEmpty()) return emptyList()
        // Batch-fetch the page's context (activity, orders, photos, runners,
        // settings, events) in six IN round-trips instead of ~8 queries per
        // row. hydrateOne's defaults keep the single-row resolve / deny /
        // escalate paths working unchanged.
        val activityByDispute = loadActivityForDisputes(disputes.map { it.id })
        val ordersById = orderRepository
            .findAllById(disputes.mapTo(mutableSetOf()) { it.orderId })
            .associateBy { it.id }
        val photosById = photoRepository
            .findAllById(disputes.mapTo(mutableSetOf()) { it.photoId })
            .associateBy { it.id }
        val runnersById = userRepository
            .findAllById(disputes.mapNotNullTo(mutableSetOf()) { it.runnerId })
            .associateBy { it.id }
        val settingsById = photographerSettingsRepository
            .findAllById(photosById.values.mapNotNullTo(mutableSetOf()) { it.photographerId })
            .associateBy { it.userId }
        val eventsById = eventRepository
            .findAllById(
                disputes.mapNotNullTo(mutableSetOf()) { d ->
                    ordersById[d.orderId]?.eventId ?: photosById[d.photoId]?.eventId
                },
            )
            .associateBy { it.id }
        return disputes.map { d ->
            val order = ordersById[d.orderId]
            val photo = photosById[d.photoId]
            hydrateOne(
                d,
                activity = activityByDispute[d.id].orEmpty(),
                order = order,
                photo = photo,
                runner = d.runnerId?.let { runnersById[it] },
                photographerSettings = photo?.photographerId?.let { settingsById[it] },
                event = (order?.eventId ?: photo?.eventId)?.let { eventsById[it] },
            )
        }
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
        order: Order? = orderRepository.findById(dispute.orderId).orElse(null),
        photo: Photo? = photoRepository.findById(dispute.photoId).orElse(null),
        runner: User? = dispute.runnerId?.let { userRepository.findById(it).orElse(null) },
        photographerSettings: PhotographerSettings? =
            photo?.photographerId?.let { photographerSettingsRepository.findById(it).orElse(null) },
        event: Event? = (order?.eventId ?: photo?.eventId)?.let { eventRepository.findById(it).orElse(null) },
    ): AdminDisputeDto {
        // A runnerId that no longer resolves still reads as "" (deleted user),
        // never as the guest-email fallback — matches the pre-batching logic.
        val runnerHandle = if (dispute.runnerId != null) {
            runner?.let { runnerDisplayHandle(it.name, it.email) } ?: ""
        } else {
            order?.let { runnerDisplayHandle("", it.recipientEmail) } ?: ""
        }

        val photographerHandle: String = photographerSettings?.handle.orEmpty()

        val eventId = order?.eventId ?: photo?.eventId ?: dispute.orderId // fallback for ghost rows
        val eventName = event?.name

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
