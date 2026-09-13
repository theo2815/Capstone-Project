package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.AdminPayoutAccountSnapshotDto
import com.quickpitik.dto.admin.AdminPayoutCycleDto
import com.quickpitik.dto.admin.AdminPayoutQrDto
import com.quickpitik.dto.admin.BulkPayoutItemResultDto
import com.quickpitik.dto.admin.BulkPayoutResultDto
import com.quickpitik.dto.admin.BulkPayoutsRequest
import com.quickpitik.dto.admin.GenerateCyclesResultDto
import com.quickpitik.dto.admin.MarkPayoutPaidRequest
import com.quickpitik.dto.earnings.buildCycleId
import com.quickpitik.entity.PayoutAccount
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.PayoutMethod
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.disbursement.DisbursementRequest
import com.quickpitik.service.disbursement.DisbursementService
import com.quickpitik.service.earnings.PayoutCycleTransitions
import com.quickpitik.service.earnings.PayoutRequestService
import com.quickpitik.service.earnings.PayoutTransition
import com.quickpitik.service.storage.StorageService
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.UUID

@Service
@Transactional
class AdminPayoutService(
    private val payoutCycleRepository: PayoutCycleRepository,
    private val payoutAccountRepository: PayoutAccountRepository,
    private val userRepository: UserRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val transactionRepository: TransactionRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val adminDecisionLogService: AdminDecisionLogService,
    private val adminDecisionLogRepository: AdminDecisionLogRepository,
    private val payoutCycleTransitions: PayoutCycleTransitions,
    private val platformProperties: PlatformProperties,
    private val disbursementService: DisbursementService,
    private val payoutRequestService: PayoutRequestService,
) {

    @Transactional(readOnly = true)
    fun list(
        statusFilter: String?,
        search: String?,
        params: PaginationParams,
    ): PaginatedResponse<AdminPayoutCycleDto> {
        val statusWire = statusFilter
            ?.takeIf { it.isNotBlank() }
            ?.let { mapAdminStatusFilter(it) }
        val searchTerm = search?.trim()?.lowercase().orEmpty()
        val page = payoutCycleRepository.pageForAdmin(statusWire, searchTerm, OffsetLimitPageable(params))
        if (page.isEmpty) return PaginatedResponse.empty(params)
        val items = hydrateMany(page.content)
        return PaginatedResponse.of(items, page.totalElements, params)
    }

    fun approve(adminId: UUID, payoutId: String): AdminPayoutCycleDto {
        val cycle = loadCycle(payoutId)
        payoutCycleTransitions.ensureAllowed(cycle, PayoutTransition.AdminApprove)
        cycle.holdReason = null
        val needsTwo = requiresTwoPerson(cycle)

        // Two-person second-approval leg (large amount, already first-approved).
        if (needsTwo && cycle.approvedBy != null) {
            if (cycle.approvedBy == adminId) {
                throw ApiException(
                    status = HttpStatus.CONFLICT,
                    code = ErrorCodes.PAYOUT_SECOND_APPROVER_REQUIRED,
                    message = "A different admin must give the second approval for this amount.",
                )
            }
            cycle.secondApprovedBy = adminId
            payoutCycleTransitions.apply(cycle, PayoutTransition.AdminApprove, adminId, "second approval")
            val decision = adminDecisionLogService.logPayoutDecision(
                adminId = adminId, targetPayoutId = cycle.id, decision = "approved-second",
            )
            adminDecisionLogService.pushMessage(
                photographerId = cycle.photographerId,
                kind = PhotographerMessageKind.PAYOUT_APPROVED,
                body = "Payout cycle ${cycle.id} was approved and is being processed.",
                sourceAdminId = adminId,
                sourceDecisionId = decision.id,
            )
            return hydrateOne(cycle)
        }

        // Two-person first-approval leg — awaits a second, different admin. The
        // cycle stays PENDING (not payable) until the second approval lands.
        if (needsTwo) {
            // Not a lifecycle move: the cycle stays PENDING and only the first
            // approver is stamped, so this deliberately does not go through
            // PayoutCycleTransitions and writes no history row.
            cycle.status = PayoutCycleStatus.PENDING
            cycle.approvedBy = adminId
            cycle.approvedAt = OffsetDateTime.now()
            payoutCycleRepository.save(cycle)
            adminDecisionLogService.logPayoutDecision(
                adminId = adminId, targetPayoutId = cycle.id, decision = "approved-first",
            )
            return hydrateOne(cycle)
        }

        // Single approval (below threshold, or fewer than two admins).
        cycle.approvedBy = adminId
        cycle.approvedAt = OffsetDateTime.now()
        payoutCycleTransitions.apply(cycle, PayoutTransition.AdminApprove, adminId)
        val decision = adminDecisionLogService.logPayoutDecision(
            adminId = adminId,
            targetPayoutId = cycle.id,
            decision = "approved",
        )
        adminDecisionLogService.pushMessage(
            photographerId = cycle.photographerId,
            kind = PhotographerMessageKind.PAYOUT_APPROVED,
            body = "Payout cycle ${cycle.id} was approved and is being processed.",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateOne(cycle)
    }

    // Payouts >= the configured threshold need a second, distinct admin — but
    // only once >= 2 admins exist, so a lone admin can still run the business.
    private fun requiresTwoPerson(cycle: PayoutCycle): Boolean {
        if (cycle.amountPhp < platformProperties.payoutTwoPersonThresholdPhp) return false
        return userRepository.countByRole(Role.ADMIN) >= 2
    }

    fun hold(adminId: UUID, payoutId: String, reason: String): AdminPayoutCycleDto {
        val cycle = loadCycle(payoutId)
        ensureManual(cycle)
        cycle.holdReason = reason.take(255)
        payoutCycleTransitions.apply(cycle, PayoutTransition.AdminHold, adminId, reason)
        val decision = adminDecisionLogService.logPayoutDecision(
            adminId = adminId,
            targetPayoutId = cycle.id,
            decision = "held",
            reason = reason,
        )
        adminDecisionLogService.pushMessage(
            photographerId = cycle.photographerId,
            kind = PhotographerMessageKind.PAYOUT_HELD,
            body = "Payout cycle ${cycle.id} was placed on hold. Reason: $reason",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateOne(cycle)
    }

    fun markPaid(adminId: UUID, payoutId: String, req: MarkPayoutPaidRequest): AdminPayoutCycleDto {
        val cycle = loadCycle(payoutId)
        ensureManual(cycle)
        payoutCycleTransitions.ensureAllowed(cycle, PayoutTransition.AdminMarkPaid)
        payoutRequestService.assertCovered(cycle)
        // AdminMarkPaid allows APPROVED/PROCESSING only — a cycle must be
        // (fully) approved before it can be paid, which also enforces two-person
        // approval for large amounts.
        cycle.paymentReference = req.paymentReference.trim()
        cycle.paidBy = adminId
        cycle.holdReason = null
        cycle.settledAt = OffsetDateTime.now()
        payoutCycleTransitions.apply(cycle, PayoutTransition.AdminMarkPaid, adminId)
        val decision = adminDecisionLogService.logPayoutDecision(
            adminId = adminId,
            targetPayoutId = cycle.id,
            decision = "marked-paid",
            meta = mapOf("paymentReference" to req.paymentReference),
        )
        adminDecisionLogService.pushMessage(
            photographerId = cycle.photographerId,
            kind = PhotographerMessageKind.PAYOUT_PAID,
            body = "Payout cycle ${cycle.id} was paid. Reference: ${req.paymentReference}.",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateOne(cycle)
    }

    // ─── Explicit lifecycle transitions (Phase 4b) ─────────────────────────
    // reject / mark-processing / mark-failed / mark-returned. Each records a
    // payout_status_history row + admin_decision_log in the same TX. Reservation
    // release is automatic: REJECTED/FAILED/RETURNED aren't in the "open" set
    // (sumReservedByPhotographer), so the funds re-enter the photographer's
    // available balance the moment the status flips.

    fun reject(adminId: UUID, payoutId: String, reason: String): AdminPayoutCycleDto {
        val cycle = loadCycle(payoutId)
        cycle.rejectedReason = reason.take(255)
        cycle.holdReason = null
        payoutCycleTransitions.apply(cycle, PayoutTransition.AdminReject, adminId, reason)
        val decision = adminDecisionLogService.logPayoutDecision(
            adminId = adminId, targetPayoutId = cycle.id, decision = "rejected", reason = reason,
        )
        adminDecisionLogService.pushMessage(
            photographerId = cycle.photographerId,
            kind = PhotographerMessageKind.ADMIN_MESSAGE,
            body = "Your payout request ${cycle.id} was rejected. Reason: $reason",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
            title = "Payout rejected",
        )
        return hydrateOne(cycle)
    }

    fun markProcessing(adminId: UUID, payoutId: String): AdminPayoutCycleDto {
        val cycle = loadCycle(payoutId)
        // Guarded before the transfer, not after: a refused transition must
        // never follow money that has already left.
        payoutCycleTransitions.ensureAllowed(cycle, PayoutTransition.AdminMarkProcessing)
        payoutRequestService.assertCovered(cycle)
        // Route the money movement through the disbursement seam. Manual = no-op
        // (admin sends by hand, then attests via mark-paid); PayMongo = creates a
        // Transfer and returns its id, with the webhook confirming later (Phase 11).
        cycle.provider = disbursementService.activeName
        disbursementRequestFor(cycle)?.let { req ->
            cycle.providerTransferId = disbursementService.initiate(req).providerTransferId
        }
        cycle.processingAt = OffsetDateTime.now()
        payoutCycleTransitions.apply(cycle, PayoutTransition.AdminMarkProcessing, adminId)
        // Internal step — no photographer inbox message; the status shows on billing.
        adminDecisionLogService.logPayoutDecision(
            adminId = adminId, targetPayoutId = cycle.id, decision = "processing",
        )
        return hydrateOne(cycle)
    }

    // Automated transfers require a frozen destination. Legacy cycles can still
    // be settled manually after the admin verifies the recipient.
    private fun disbursementRequestFor(cycle: PayoutCycle): DisbursementRequest? {
        val method = cycle.accountSnapshotMethod
        val number = cycle.accountSnapshotNumber
        val name = cycle.accountSnapshotName
        if (method != null && number != null && name != null) {
            return DisbursementRequest(cycle.id, cycle.amountPhp, method, number, name)
        }
        if (cycle.provider != "manual") throw ApiException(
            HttpStatus.CONFLICT, ErrorCodes.INVALID_STATE_TRANSITION,
            "This legacy payout has no verified destination snapshot. Settle it manually.",
        )
        return null
    }

    fun markFailed(adminId: UUID, payoutId: String, reason: String): AdminPayoutCycleDto {
        val cycle = loadCycle(payoutId)
        ensureManual(cycle)
        cycle.failedReason = reason.take(255)
        payoutCycleTransitions.apply(cycle, PayoutTransition.AdminMarkFailed, adminId, reason)
        val decision = adminDecisionLogService.logPayoutDecision(
            adminId = adminId, targetPayoutId = cycle.id, decision = "failed", reason = reason,
        )
        adminDecisionLogService.pushMessage(
            photographerId = cycle.photographerId,
            kind = PhotographerMessageKind.ADMIN_MESSAGE,
            body = "Your payout ${cycle.id} could not be sent (no money left your side). Reason: $reason. It can be retried.",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
            title = "Payout failed",
        )
        return hydrateOne(cycle)
    }

    fun markReturned(adminId: UUID, payoutId: String, reason: String): AdminPayoutCycleDto {
        val cycle = loadCycle(payoutId)
        ensureManual(cycle)
        cycle.returnedAt = OffsetDateTime.now()
        payoutCycleTransitions.apply(cycle, PayoutTransition.AdminMarkReturned, adminId, reason)
        val decision = adminDecisionLogService.logPayoutDecision(
            adminId = adminId, targetPayoutId = cycle.id, decision = "returned", reason = reason,
        )
        adminDecisionLogService.pushMessage(
            photographerId = cycle.photographerId,
            kind = PhotographerMessageKind.ADMIN_MESSAGE,
            body = "Your payout ${cycle.id} was returned by the bank/e-wallet. Reason: $reason. Please check your payout account.",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
            title = "Payout returned",
        )
        return hydrateOne(cycle)
    }

    private fun ensureManual(cycle: PayoutCycle) {
        if (cycle.provider != "manual") {
            throw ApiException(
                HttpStatus.CONFLICT, ErrorCodes.INVALID_STATE_TRANSITION,
                "This transfer is managed by the payment provider. Wait for its confirmed outcome.",
            )
        }
    }

    /**
     * POST /api/v1/admin/payouts/bulk (C-3).
     *
     * Idempotent on the inbound `Idempotency-Key` header. A retry with the
     * same (admin_id, idempotency_key) returns the original group_id and
     * does NOT re-issue per-cycle audit-log rows or photographer-inbox
     * messages — every per-item write is gated by the partial unique index
     * `uq_admin_decision_log_admin_idem_target_payout` so duplicates are
     * locked out at the DB even under concurrent retries.
     *
     * Replays are checked under the lock before any state mutation. Errors raised by
     * per-item validation (already-paid, missing-reason, unknown-action)
     * are caught per-row and reported in the result list, exactly as in
     * the non-replay path.
     */
    fun bulk(adminId: UUID, idempotencyKey: String, req: BulkPayoutsRequest): BulkPayoutResultDto {
        userRepository.lockById(adminId)
        // Fast path: a prior bulk run with this (admin_id, idempotency_key)
        // already minted a group_id — reuse it so the dashboard kpi
        // (`COUNT(DISTINCT group_id)`) still collapses retries into one
        // logical decision and the FE bulk-undo button has a stable id.
        val groupId = adminDecisionLogRepository.findGroupIdByAdminAndIdempotencyKey(
            adminId,
            idempotencyKey,
        ) ?: UUID.randomUUID()

        val results = try {
            req.ids.distinct().sorted().map { id ->
                try {
                    when (req.action.trim().lowercase()) {
                        "approve" -> {
                            bulkApprove(adminId, id, groupId, idempotencyKey)
                            BulkPayoutItemResultDto(id = id, ok = true)
                        }
                        "hold" -> {
                            val reason = req.reason
                                ?: throw ValidationException(
                                    code = ErrorCodes.VALIDATION_ERROR,
                                    message = "reason required for hold",
                                    field = "reason",
                                )
                            bulkHold(adminId, id, reason, groupId, idempotencyKey)
                            BulkPayoutItemResultDto(id = id, ok = true)
                        }
                        else -> throw ValidationException(
                            code = ErrorCodes.INVALID_BULK_ACTION,
                            message = "action must be approve or hold",
                            field = "action",
                        )
                    }
                } catch (ex: ValidationException) {
                    BulkPayoutItemResultDto(id = id, ok = false, error = ex.message)
                } catch (ex: NotFoundException) {
                    BulkPayoutItemResultDto(id = id, ok = false, error = ex.message)
                } catch (ex: ApiException) {
                    BulkPayoutItemResultDto(id = id, ok = false, error = ex.message)
                }
            }
        } catch (ex: DataIntegrityViolationException) {
            // Concurrent retry with the same Idempotency-Key hit the partial
            // unique on (admin_id, idempotency_key, target_payout_id) before
            // the in-flight request committed. Postgres aborts the parent TX
            // — Spring rolls it back when this throw escapes. The FE's next
            // retry sees committed rows via the existsBy* fast path inside
            // bulkApprove / bulkHold and short-circuits to success. Mirrors
            // the OrderService.create:159 DIE-replay-and-return shape, but
            // signals retry rather than re-reading because the bulk surface
            // has no canonical "primary order" to fold the response onto.
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.IDEMPOTENT_REPLAY_IN_PROGRESS,
                message = "An identical bulk operation is in progress. Retry shortly.",
            )
        }
        return BulkPayoutResultDto(groupId = groupId, results = results)
    }

    private fun bulkApprove(adminId: UUID, payoutId: String, groupId: UUID, idempotencyKey: String) {
        val cycle = loadCycle(payoutId)
        if (adminDecisionLogRepository.existsByAdminIdAndIdempotencyKeyAndTargetPayoutId(
                adminId, idempotencyKey, payoutId,
            )) return
        payoutCycleTransitions.ensureAllowed(cycle, PayoutTransition.AdminApprove)
        val needsTwo = requiresTwoPerson(cycle)
        if (needsTwo && cycle.approvedBy == adminId) throw ApiException(
            HttpStatus.CONFLICT, ErrorCodes.PAYOUT_SECOND_APPROVER_REQUIRED,
            "A different admin must give the second approval for this amount.",
        )
        cycle.holdReason = null
        val reachedApproved = when {
            needsTwo && cycle.approvedBy != null && cycle.approvedBy != adminId -> {
                cycle.secondApprovedBy = adminId
                true
            }
            needsTwo && cycle.approvedBy == null -> {
                // Mirrors approve()'s first leg: stays PENDING, no history row.
                cycle.status = PayoutCycleStatus.PENDING
                cycle.approvedBy = adminId
                cycle.approvedAt = OffsetDateTime.now()
                false
            }
            needsTwo -> false // same admin already first-approved (or retry) — hold at PENDING
            else -> {
                cycle.approvedBy = adminId
                cycle.approvedAt = OffsetDateTime.now()
                true
            }
        }
        if (reachedApproved) {
            payoutCycleTransitions.apply(cycle, PayoutTransition.AdminApprove, adminId)
        } else {
            payoutCycleRepository.save(cycle)
        }
        val decision = adminDecisionLogService.logPayoutDecision(
            adminId = adminId,
            targetPayoutId = cycle.id,
            decision = if (reachedApproved) "approved" else "approved-first",
            groupId = groupId,
            idempotencyKey = idempotencyKey,
        )
        if (reachedApproved) {
            adminDecisionLogService.pushMessage(
                photographerId = cycle.photographerId,
                kind = PhotographerMessageKind.PAYOUT_APPROVED,
                body = "Payout cycle ${cycle.id} was approved.",
                sourceAdminId = adminId,
                sourceDecisionId = decision.id,
            )
        }
    }

    private fun bulkHold(
        adminId: UUID,
        payoutId: String,
        reason: String,
        groupId: UUID,
        idempotencyKey: String,
    ) {
        val cycle = loadCycle(payoutId)
        if (adminDecisionLogRepository.existsByAdminIdAndIdempotencyKeyAndTargetPayoutId(
                adminId, idempotencyKey, payoutId,
            )) return
        ensureManual(cycle)
        cycle.holdReason = reason.take(255)
        payoutCycleTransitions.apply(cycle, PayoutTransition.AdminHold, adminId, reason)
        val decision = adminDecisionLogService.logPayoutDecision(
            adminId = adminId,
            targetPayoutId = cycle.id,
            decision = "held",
            reason = reason,
            groupId = groupId,
            idempotencyKey = idempotencyKey,
        )
        adminDecisionLogService.pushMessage(
            photographerId = cycle.photographerId,
            kind = PhotographerMessageKind.PAYOUT_HELD,
            body = "Payout cycle ${cycle.id} was placed on hold. Reason: $reason",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
    }

    /**
     * Admin-triggered cycle generator. Scans `transactions` for the given ISO
     * week (Mon→Sun in Asia/Manila), groups by photographer, sums kept amount
     * (refund rows are negative so they net out), counts non-refund sales, and
     * upserts a PayoutCycle row per eligible photographer in pending_review
     * (DB PENDING, settledAt null).
     *
     * Idempotent: re-running for the same week refreshes pending-review rows
     * in place. Rows that already carry an admin decision (paid, held,
     * approved, or awaiting a second approval) are skipped so the generator cannot clobber
     * human judgment.
     *
     * Eligibility: any photographer with net amount > ₱0 in the window. The
     * cycle's payout method is the photographer's primary account method, or
     * GCASH as a safe default when no account is set — the admin still sees
     * an empty payoutAccount snapshot in the queue and can hold for setup.
     *
     * **Known race, accepted (documented 2026-08-16).** The aggregate below and
     * the per-row upsert are two statements, so a refund committing between
     * them lands in neither: the summed amount is already stale and the cycle
     * is written from it, with nothing recording the discrepancy.
     *
     * Left as-is deliberately. This path is an optional backfill superseded by
     * the request-based flow (`PayoutRequestService`, 2026-05-19) — the normal
     * way a cycle is created is a photographer asking for one, which reads the
     * balance inside a single transaction. Raising this method to SERIALIZABLE
     * (or locking the transaction window) would add contention to the whole
     * payout path to protect a route that runs rarely and by hand.
     *
     * The mitigation is operational, not structural: re-running for the same
     * week refreshes any still-pending_review row in place, so a mis-summed
     * cycle self-corrects on the next run as long as no admin has decided it
     * yet. If this path ever becomes automatic (a cron), revisit — an
     * unattended generator has no human to notice the discrepancy.
     */
    fun generateForWeek(weekOf: LocalDate): GenerateCyclesResultDto {
        if (weekOf.dayOfWeek != java.time.DayOfWeek.MONDAY) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "weekOf must be a Monday (ISO week start)",
                field = "weekOf",
            )
        }

        val zone = ZoneId.of("Asia/Manila")
        val from = weekOf.atStartOfDay(zone).toOffsetDateTime()
        val to = weekOf.plusDays(7).atStartOfDay(zone).toOffsetDateTime()

        val rows = transactionRepository.aggregateByPhotographerInWindow(from, to)

        var created = 0
        var updated = 0
        var skipped = 0
        var totalAmount = BigDecimal.ZERO

        for (row in rows.sortedBy { it[0].toString() }) {
            val photographerId = row[0] as UUID
            userRepository.lockById(photographerId)
            val amount = row[1] as BigDecimal
            val itemCount = (row[2] as Number).toInt()

            val settings = photographerSettingsRepository.findById(photographerId).orElse(null)
            val handle = settings?.handle
            val cycleId = buildCycleId(weekOf, handle)

            val primaryAccount = payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId)
            val method = primaryAccount?.method ?: PayoutMethod.GCASH

            val existing = payoutCycleRepository.lockById(cycleId)

            when {
                existing == null -> {
                    val cycle = PayoutCycle(
                        id = cycleId,
                        photographerId = photographerId,
                        weekOf = weekOf,
                        methodWire = method.wire,
                        accountSnapshotMethod = primaryAccount?.method?.wire,
                        accountSnapshotNumber = primaryAccount?.accountNumber,
                        accountSnapshotName = primaryAccount?.accountName,
                        accountSnapshotQrKey = primaryAccount?.qrS3Key,
                    )
                    cycle.amountPhp = amount
                    cycle.itemCount = itemCount
                    cycle.method = method
                    cycle.status = PayoutCycleStatus.PENDING
                    cycle.settledAt = null
                    payoutCycleRepository.save(cycle)
                    created++
                    totalAmount = totalAmount.add(amount)
                }
                existing.status == PayoutCycleStatus.SCHEDULED ||
                    (existing.status == PayoutCycleStatus.PENDING && existing.approvedBy == null) -> {
                    // Only refresh a truly undecided cycle (scheduled, or
                    // pending-review with no review yet). Any decided/terminal
                    // state (approved/processing/paid/held/failed/returned/
                    // rejected/cancelled) is left alone — re-summing would
                    // silently rewrite an amount the admin already acted on.
                    existing.amountPhp = amount
                    existing.itemCount = itemCount
                    existing.methodWire = existing.accountSnapshotMethod ?: method.wire
                    existing.status = PayoutCycleStatus.PENDING
                    existing.settledAt = null
                    payoutCycleRepository.save(existing)
                    updated++
                    totalAmount = totalAmount.add(amount)
                }
                else -> {
                    skipped++
                }
            }
        }

        return GenerateCyclesResultDto(
            weekOf = weekOf.toString(),
            created = created,
            updated = updated,
            skipped = skipped,
            totalAmount = totalAmount,
        )
    }

    // ─── Helpers ──────────────────────────────────────────────────────────
    private fun loadCycle(payoutId: String): PayoutCycle =
        payoutCycleRepository.lockById(payoutId) ?: throw
            NotFoundException(code = ErrorCodes.PAYOUT_NOT_FOUND, message = "Payout cycle not found")

    // Approval is a stored state. Only pending_review needs a wire alias.
    private fun mapAdminStatusFilter(filter: String): String? {
        return when (filter.trim().lowercase()) {
            "pending_review", "pending" -> PayoutCycleStatus.PENDING.wire
            "approved" -> PayoutCycleStatus.APPROVED.wire
            "processing" -> PayoutCycleStatus.PROCESSING.wire
            "held" -> PayoutCycleStatus.HELD.wire
            "paid" -> PayoutCycleStatus.PAID.wire
            "failed" -> PayoutCycleStatus.FAILED.wire
            "returned" -> PayoutCycleStatus.RETURNED.wire
            "rejected" -> PayoutCycleStatus.REJECTED.wire
            "cancelled" -> PayoutCycleStatus.CANCELLED.wire
            "scheduled" -> PayoutCycleStatus.SCHEDULED.wire
            else -> null
        }
    }

    private fun hydrateMany(cycles: List<PayoutCycle>): List<AdminPayoutCycleDto> {
        if (cycles.isEmpty()) return emptyList()
        // Batch-prefetch the page's users / settings / payout accounts in three
        // IN round-trips instead of 3–4 queries per row (same shape as
        // OrderService.hydrateList). hydrateOne's defaults keep the single-row
        // approve/hold/pay paths working unchanged.
        val photographerIds = cycles.mapTo(mutableSetOf()) { it.photographerId }
        val usersById = userRepository.findAllById(photographerIds).associateBy { it.id }
        val settingsById = photographerSettingsRepository.findAllById(photographerIds).associateBy { it.userId }
        val accountsByUser = payoutAccountRepository
            .findAllByUserIdInOrderByCreatedAtAsc(photographerIds)
            .groupBy { it.userId }
        return cycles.map { cycle ->
            val candidates = accountsByUser[cycle.photographerId].orEmpty()
            hydrateOne(
                cycle,
                user = usersById[cycle.photographerId],
                settings = settingsById[cycle.photographerId],
                account = candidates.firstOrNull { it.isPrimary } ?: candidates.firstOrNull(),
            )
        }
    }

    internal fun hydrateOne(
        cycle: PayoutCycle,
        user: User? = userRepository.findById(cycle.photographerId).orElse(null),
        settings: PhotographerSettings? = photographerSettingsRepository.findById(cycle.photographerId).orElse(null),
        account: PayoutAccount? = payoutAccountRepository.findByUserIdAndIsPrimaryTrue(cycle.photographerId)
            ?: payoutAccountRepository.findAllByUserIdOrderByCreatedAtAsc(cycle.photographerId).firstOrNull(),
    ): AdminPayoutCycleDto {
        return AdminPayoutCycleDto(
            id = cycle.id,
            photographerId = cycle.photographerId,
            photographerName = user?.name ?: "—",
            brandName = settings?.brandName,
            handle = settings?.handle,
            weekOf = cycle.weekOf.toString(),
            amount = cycle.amountPhp,
            itemCount = cycle.itemCount,
            method = cycle.methodWire,
            status = toAdminWireStatus(cycle),
            submittedAt = cycle.createdAt.toString(),
            reviewedAt = cycle.approvedAt?.toString(),
            paidAt = if (cycle.status == PayoutCycleStatus.PAID) cycle.settledAt?.toString() else null,
            paymentReference = cycle.paymentReference,
            holdReason = cycle.holdReason,
            rejectedReason = cycle.rejectedReason,
            failedReason = cycle.failedReason,
            approvedBy = cycle.approvedBy?.toString(),
            secondApprovedBy = cycle.secondApprovedBy?.toString(),
            provider = cycle.provider,
            payoutAccount = snapshotAccount(cycle, account),
        )
    }

    private fun toAdminWireStatus(cycle: PayoutCycle): String {
        return when (cycle.status) {
            PayoutCycleStatus.PENDING -> "pending_review"
            PayoutCycleStatus.SCHEDULED -> "pending_review"
            PayoutCycleStatus.APPROVED -> "approved"
            PayoutCycleStatus.PROCESSING -> "processing"
            PayoutCycleStatus.HELD -> "held"
            PayoutCycleStatus.PAID -> "paid"
            PayoutCycleStatus.FAILED -> "failed"
            PayoutCycleStatus.RETURNED -> "returned"
            PayoutCycleStatus.REJECTED -> "rejected"
            PayoutCycleStatus.CANCELLED -> "cancelled"
        }
    }

    // Prefer the cycle's immutable destination snapshot (where the money will
    // actually go) so a later payout-account change can't make the admin send to
    // a new number; fall back to the live account for legacy cycles without a
    // snapshot. Only the captured QR is safe to show beside a frozen number.
    private fun snapshotAccount(
        cycle: PayoutCycle,
        account: PayoutAccount?,
    ): AdminPayoutAccountSnapshotDto {
        val method = cycle.accountSnapshotMethod ?: account?.method?.wire ?: ""
        val number = cycle.accountSnapshotNumber ?: account?.accountNumber ?: ""
        val name = cycle.accountSnapshotName ?: account?.accountName ?: ""
        val qr = cycle.accountSnapshotQrKey?.let { key ->
            AdminPayoutQrDto(
                // Reuse the cover TTL — both are private images surfaced via
                // a presigned GET that the FE renders straight into <img src>.
                dataUrl = storageService.presignedGetUrl(key, storageProperties.presignedTtl.cover),
                uploadedAt = cycle.createdAt.toString(),
            )
        }
        return AdminPayoutAccountSnapshotDto(
            method = method,
            accountNumber = number,
            accountName = name,
            qr = qr,
        )
    }
}
