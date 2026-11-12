package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.DisputeQueueProperties
import com.quickpitik.config.PaymongoProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.AdminDisputeDto
import com.quickpitik.dto.admin.DisputeActivityEntry
import com.quickpitik.dto.admin.DisputeDenyReasonDto
import com.quickpitik.dto.admin.DisputeOrderSnapshotDto
import com.quickpitik.dto.admin.DisputePhotoSnapshotDto
import com.quickpitik.dto.admin.ResolveDisputeRequest
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeResolution
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.Event
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.Payment
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.RunnerMessageKind
import com.quickpitik.entity.User
import com.quickpitik.entity.refundNeedsReconciliation
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.service.orders.PaymongoRefundService
import com.quickpitik.service.orders.refundBlocker
import com.quickpitik.service.orders.refundCancelsAt
import com.quickpitik.service.orders.refundLinkExpiresAt
import com.quickpitik.service.orders.refundWindowClosesAt
import com.quickpitik.service.orders.settledPaymongo
import com.quickpitik.service.storage.StorageService
import org.springframework.data.domain.Pageable
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID

// Status lists the list views pass to DisputeRepository.pageForAdmin (an empty IN list matches nothing).
private val UNRESOLVED_STATUS_WIRES = listOf(DisputeStatus.OPEN.wire, DisputeStatus.ESCALATED.wire)
private val CLOSED_STATUS_WIRES = listOf(DisputeStatus.RESOLVED.wire, DisputeStatus.DENIED.wire, DisputeStatus.WITHDRAWN.wire)
private val ALL_STATUS_WIRES = UNRESOLVED_STATUS_WIRES + CLOSED_STATUS_WIRES

@Service
@Transactional
class AdminDisputeService(
    private val disputeRepository: DisputeRepository,
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val paymentRepository: PaymentRepository,
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
    private val paymongoProperties: PaymongoProperties,
    private val disputeQueueProperties: DisputeQueueProperties,
    private val clock: Clock,
) {

    @Transactional(readOnly = true)
    fun get(disputeId: UUID): AdminDisputeDto {
        val dispute = disputeRepository.findById(disputeId).orElseThrow {
            NotFoundException(code = ErrorCodes.DISPUTE_NOT_FOUND, message = "Dispute not found")
        }
        return hydrateOne(dispute)
    }

    // Views (plan 5.2): none, `all` and `closed` page in SQL, newest first; `needs_me`, `overdue`
    // and `refund_run` read the attention of every unresolved dispute and sort by due time.
    @Transactional(readOnly = true)
    fun list(
        view: String?,
        statusFilter: String?,
        query: String?,
        params: PaginationParams,
    ): PaginatedResponse<AdminDisputeDto> {
        val viewWire = view?.trim()?.lowercase()?.takeIf { it.isNotEmpty() }
        val statusWire = statusFilter?.takeIf { it.isNotBlank() }?.let { parseStatus(it).wire }
        if (viewWire != null && statusWire != null) {
            throw ValidationException(message = "Use view or status, not both", field = "view")
        }
        // shortcut: LIKE wildcards in q are left as-is and only the length is capped (admin-only search); escape them if a non-admin search reuses this.
        val q = query?.trim()?.take(100) ?: ""
        val keep: (DisputeQueueState) -> Boolean = when (viewWire) {
            null, "all" -> return sqlPage(statusWire?.let { listOf(it) } ?: ALL_STATUS_WIRES, q, params)
            "closed" -> return sqlPage(CLOSED_STATUS_WIRES, q, params)
            "needs_me" -> { s -> s.attention.needsAdmin }
            "overdue" -> { s -> s.attention.needsAdmin && s.overdue }
            "refund_run" -> { s -> s.attention in PARKED_QRPH_ATTENTIONS }
            else -> throw ValidationException(
                message = "view must be one of needs_me / overdue / refund_run / closed / all",
                field = "view",
            )
        }
        val now = OffsetDateTime.now(clock)
        val sorted = unresolvedWithState(q, now)
            .filter { keep(it.second) }
            .sortedWith(compareBy({ it.second.dueAt }, { it.first.openedAt }, { it.first.id }))
        val page = sorted.drop(params.offset).take(params.limit).map { it.first }
        return PaginatedResponse.of(hydrateMany(page, now), sorted.size.toLong(), params)
    }

    // The badge's numbers: the same load and classify as the `needs_me` view.
    @Transactional(readOnly = true)
    fun queueCounts(): DisputeQueueCounts {
        val states = unresolvedWithState("", OffsetDateTime.now(clock)).map { it.second }.filter { it.attention.needsAdmin }
        return DisputeQueueCounts(needsAdmin = states.size.toLong(), overdue = states.count { it.overdue }.toLong())
    }

    private fun sqlPage(statusWires: List<String>, q: String, params: PaginationParams): PaginatedResponse<AdminDisputeDto> {
        val page = disputeRepository.pageForAdmin(statusWires, q, OffsetLimitPageable(params))
        if (page.isEmpty) return PaginatedResponse.empty(params)
        return PaginatedResponse.of(hydrateMany(page.content), page.totalElements, params)
    }

    // shortcut: sorts unresolved disputes in memory; fine while open plus escalated stay in the low thousands (DW-0.2 count); upgrade to a stored due_at column with the photo-removal sweep recipe if it grows
    private fun unresolvedWithState(q: String, now: OffsetDateTime): List<Pair<Dispute, DisputeQueueState>> {
        val disputes = disputeRepository.pageForAdmin(UNRESOLVED_STATUS_WIRES, q, Pageable.unpaged()).content
        if (disputes.isEmpty()) return emptyList()
        val ordersById = orderRepository.findAllById(disputes.mapTo(mutableSetOf()) { it.orderId }).associateBy { it.id }
        return disputes.map { it to queueState(it, ordersById[it.orderId], paymongoProperties, disputeQueueProperties, now) }
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

    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    fun sendRefundLink(adminId: UUID, disputeId: UUID, url: String): AdminDisputeDto {
        paymongoRefundService.sendRefundLink(adminId, disputeId, url)
        return transactionTemplate.execute {
            val dispute = disputeRepository.findById(disputeId).orElseThrow {
                NotFoundException(code = ErrorCodes.DISPUTE_NOT_FOUND, message = "Dispute not found")
            }
            hydrateOne(dispute)
        } ?: error("Refund link returned null")
    }

    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    fun markRefunded(adminId: UUID, disputeId: UUID): AdminDisputeDto {
        paymongoRefundService.markRefunded(adminId, disputeId)
        return transactionTemplate.execute {
            val dispute = disputeRepository.findById(disputeId).orElseThrow {
                NotFoundException(code = ErrorCodes.DISPUTE_NOT_FOUND, message = "Dispute not found")
            }
            hydrateOne(dispute)
        } ?: error("Mark refunded returned null")
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
     *
     * DW-4: a `reasonCode` (DisputeDenyReason) is stored as its runner text, then
     * "Note from our team: <note>" when a note is given; `other` stores the note alone.
     */
    fun deny(adminId: UUID, disputeId: UUID, reason: String?, reasonCode: String? = null, note: String? = null): AdminDisputeDto {
        val code = reasonCode?.trim()?.takeIf { it.isNotEmpty() }
        val written = reason?.trim()?.takeIf { it.isNotEmpty() }
        val noteText = note?.trim()?.takeIf { it.isNotEmpty() }
        val denyReason = code?.let {
            DisputeDenyReason.fromCode(it) ?: throw ValidationException(message = "Unknown deny reason", field = "reasonCode")
        }
        val text: String? = if (denyReason == null) {
            if (noteText != null) throw ValidationException(message = "A note needs a deny reason code", field = "note")
            written
        } else {
            if (written != null) {
                throw ValidationException(message = "Send a deny reason code or a written reason, not both", field = "reason")
            }
            when {
                denyReason.noteRequired ->
                    noteText ?: throw ValidationException(message = "Write the reason the runner will read", field = "note")
                noteText == null -> denyReason.runnerText
                else -> "${denyReason.runnerText} Note from our team: $noteText"
            }
        }
        val dispute = disputeRepository.findByIdForUpdate(disputeId) ?: throw
            NotFoundException(code = ErrorCodes.DISPUTE_NOT_FOUND, message = "Dispute not found")
        if (dispute.refundNeedsReconciliation()) {
            // A parked QR Ph refund whose link was sent: the refund service checks the time and needs a reason.
            if (dispute.refundLinkSentAt != null) {
                // A policy reason would read wrongly about an approved refund; only a stale tab sends one here.
                if (denyReason != null) {
                    throw ApiException(
                        status = HttpStatus.CONFLICT,
                        code = ErrorCodes.INVALID_STATE_TRANSITION,
                        message = "This refund was approved and its link sent. Refresh to see it.",
                    )
                }
                paymongoRefundService.cancelUnclaimedRefund(adminId, disputeId, text)
                return hydrateOne(dispute)
            }
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Refund is in progress or requires manual reconciliation",
            )
        }
        if (dispute.status == DisputeStatus.RESOLVED || dispute.status == DisputeStatus.DENIED ||
            dispute.status == DisputeStatus.WITHDRAWN
        ) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Dispute is already ${dispute.status.wire}",
            )
        }
        dispute.status = DisputeStatus.DENIED
        dispute.resolution = DisputeResolution.DENY
        dispute.refundAmountPhp = null
        dispute.resolvedAt = OffsetDateTime.now(clock)
        disputeRepository.save(dispute)

        val photographerId = photoRepository.findById(dispute.photoId).map { it.photographerId }.orElse(null)
        val decision = adminDecisionLogService.logDisputeDecision(
            adminId = adminId,
            targetDisputeId = dispute.id,
            decision = "denied",
            reason = text,
            meta = denyReason?.let { mapOf("reasonCode" to it.code) },
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
                (text?.let { " Reason: $it" } ?: ""),
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateOne(dispute)
    }

    // Every deny reason (plan 5.4); a fixed list, so no transaction.
    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    fun denyReasons(): PaginatedResponse<DisputeDenyReasonDto> {
        val items = DisputeDenyReason.entries.map { DisputeDenyReasonDto(it.code, it.adminLabel, it.runnerText, it.noteRequired) }
        return PaginatedResponse.of(items, items.size.toLong(), PaginationParams.of(0, items.size))
    }

    fun escalate(adminId: UUID, disputeId: UUID, reason: String?): AdminDisputeDto {
        val dispute = disputeRepository.findByIdForUpdate(disputeId) ?: throw
            NotFoundException(code = ErrorCodes.DISPUTE_NOT_FOUND, message = "Dispute not found")
        if (dispute.refundNeedsReconciliation()) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Refund is in progress or requires manual reconciliation",
            )
        }
        if (dispute.status != DisputeStatus.OPEN) {
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
    private fun hydrateMany(disputes: List<Dispute>, now: OffsetDateTime = OffsetDateTime.now(clock)): List<AdminDisputeDto> {
        if (disputes.isEmpty()) return emptyList()
        // Batch-fetch the page's context (activity, orders, photos, runners,
        // settings, events) in six IN round-trips instead of ~8 queries per
        // row. hydrateOne's defaults keep the single-row resolve / deny /
        // escalate paths working unchanged.
        val activityByDispute = loadActivityForDisputes(disputes.map { it.id })
        val ordersById = orderRepository
            .findAllById(disputes.mapTo(mutableSetOf()) { it.orderId })
            .associateBy { it.id }
        val itemsById = orderItemRepository
            .findByIdOrderIdIn(disputes.mapTo(mutableSetOf()) { it.orderId })
            .associateBy { it.id }
        val paymentsByOrder = paymentRepository
            .findByOrderIdIn(disputes.mapTo(mutableSetOf()) { it.orderId })
            .groupBy { it.orderId }
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
        val checkoutByRef = loadCheckouts(
            disputes.filter { it.refundabilityShown() }
                .mapNotNullTo(mutableSetOf()) { paymentsByOrder[it.orderId].orEmpty().settledPaymongo()?.providerRef },
        )
        // One clock read per page (the caller's `now`), so no row crosses a deadline mid-page.
        return disputes.map { d ->
            val order = ordersById[d.orderId]
            val photo = photosById[d.photoId]
            hydrateOne(
                d,
                activity = activityByDispute[d.id].orEmpty(),
                order = order,
                orderItem = itemsById[OrderItemId(d.orderId, d.photoId)],
                payment = paymentsByOrder[d.orderId].orEmpty().settledPaymongo(),
                photo = photo,
                runner = d.runnerId?.let { runnersById[it] },
                photographerSettings = photo?.photographerId?.let { settingsById[it] },
                event = (order?.eventId ?: photo?.eventId)?.let { eventsById[it] },
                now = now,
                checkoutByRef = checkoutByRef,
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
        orderItem: OrderItem? = orderItemRepository.findById(OrderItemId(dispute.orderId, dispute.photoId)).orElse(null),
        payment: Payment? = paymentRepository.findByOrderId(dispute.orderId).settledPaymongo(),
        photo: Photo? = photoRepository.findById(dispute.photoId).orElse(null),
        runner: User? = dispute.runnerId?.let { userRepository.findById(it).orElse(null) },
        photographerSettings: PhotographerSettings? =
            photo?.photographerId?.let { photographerSettingsRepository.findById(it).orElse(null) },
        event: Event? = (order?.eventId ?: photo?.eventId)?.let { eventRepository.findById(it).orElse(null) },
        now: OffsetDateTime = OffsetDateTime.now(clock),
        // null: load this one dispute's checkout if its refundability is shown.
        checkoutByRef: Map<String, CheckoutFacts>? = null,
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
        val queue = queueState(dispute, order, paymongoProperties, disputeQueueProperties, now)
        val parkedQrph = order?.takeIf { queue.attention in PARKED_QRPH_ATTENTIONS }
        val itemCharge = orderItem?.pricePhpAtPurchase?.subtract(orderItem.discountPhp)
        val shown = itemCharge != null && dispute.refundabilityShown()
        val checkout = payment?.providerRef?.takeIf { shown }
            ?.let { ref -> (checkoutByRef ?: loadCheckouts(setOf(ref)))[ref] }
        // The same rule `prepare` refuses with, for the photo's full charge (plan 5.3).
        val blocker = itemCharge?.takeIf { shown }
            ?.let { paymongoProperties.refundBlocker(order, payment, checkout?.total, it, now) }

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
            refundStatus = dispute.refundStatus,
            paymentId = payment?.providerPaymentId,
            paymentRef = payment?.providerRef,
            refundLinkUrl = dispute.refundLinkUrl,
            refundLinkSentAt = dispute.refundLinkSentAt?.toString(),
            refundLinkExpiresAt = parkedQrph?.let { paymongoProperties.refundLinkExpiresAt(dispute)?.toString() },
            refundWindowClosesAt = parkedQrph?.let { paymongoProperties.refundWindowClosesAt(it).toString() },
            refundCancelsAt = parkedQrph?.let { paymongoProperties.refundCancelsAt(it, dispute).toString() },
            itemChargeAmount = itemCharge,
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
                    val key = it.watermarkS3Key ?: it.thumbnailS3Key ?: it.s3Key
                    storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
                },
            ),
            activity = activity,
            refundRequestedAt = dispute.refundRequestedAt?.toString(),
            attention = queue.attention.wire,
            dueAt = queue.dueAt?.toString(),
            overdue = queue.overdue,
            linkExpiringSoon = queue.linkExpiringSoon,
            refundable = if (shown) blocker == null else null,
            refundBlockedReason = blocker?.wire,
            refundBlockedMessage = blocker?.message,
            paymentPhotoCount = checkout?.photoCount,
        )
    }

    // A refund can still be approved: undecided, and no refund under way (a failed one can be tried again).
    private fun Dispute.refundabilityShown(): Boolean =
        (status == DisputeStatus.OPEN || status == DisputeStatus.ESCALATED) &&
            (refundStatus == null || refundStatus == "failed")

    internal data class CheckoutFacts(val total: BigDecimal, val photoCount: Int)

    // Per PayMongo reference: the sum of every payment sharing it, whatever its status (as `prepare` sums it),
    // and the photos on those payments' orders. Two queries for the whole page.
    private fun loadCheckouts(refs: Set<String>): Map<String, CheckoutFacts> {
        if (refs.isEmpty()) return emptyMap()
        val payments = paymentRepository.findAllByProviderAndProviderRefIn("paymongo", refs)
        val orderIds = payments.mapTo(mutableSetOf()) { it.orderId }
        val photosByOrder = if (orderIds.isEmpty()) emptyMap() else
            orderItemRepository.findByIdOrderIdIn(orderIds).groupingBy { it.id.orderId }.eachCount()
        return payments.groupBy { it.providerRef!! }.mapValues { (_, shared) ->
            CheckoutFacts(
                total = shared.fold(BigDecimal.ZERO) { sum, it -> sum + it.amountPhp },
                photoCount = shared.mapTo(mutableSetOf()) { it.orderId }.sumOf { photosByOrder[it] ?: 0 },
            )
        }
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
