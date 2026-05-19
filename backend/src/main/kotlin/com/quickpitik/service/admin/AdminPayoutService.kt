package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
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
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
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
        // approve = surface as "approved" without actually paying out yet.
        // Mirror the FE state machine: pending_review → approved → paid.
        // DB stays at PENDING; the admin-facing wire form is computed from
        // `settledAt + paymentReference + holdReason` — see toAdminWireStatus.
        if (cycle.status == PayoutCycleStatus.PAID) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Cycle is already paid",
            )
        }
        // Use settledAt as "reviewedAt" once the admin approves, with no
        // paymentReference until mark-paid lands.
        cycle.holdReason = null
        cycle.status = PayoutCycleStatus.PENDING
        cycle.settledAt = cycle.settledAt ?: OffsetDateTime.now()
        payoutCycleRepository.save(cycle)
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

    fun hold(adminId: UUID, payoutId: String, reason: String): AdminPayoutCycleDto {
        val cycle = loadCycle(payoutId)
        if (cycle.status == PayoutCycleStatus.PAID) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Cannot hold a paid cycle",
            )
        }
        cycle.status = PayoutCycleStatus.HELD
        cycle.holdReason = reason.take(255)
        payoutCycleRepository.save(cycle)
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
        if (cycle.status == PayoutCycleStatus.PAID) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Cycle is already paid",
            )
        }
        cycle.status = PayoutCycleStatus.PAID
        cycle.paymentReference = req.paymentReference.trim()
        cycle.holdReason = null
        cycle.settledAt = OffsetDateTime.now()
        payoutCycleRepository.save(cycle)
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
     * State mutations on the cycle (status / holdReason / settledAt) are
     * applied unconditionally — they're idempotent under the same final
     * state, so re-applying them on retry is a no-op. Errors raised by
     * per-item validation (already-paid, missing-reason, unknown-action)
     * are caught per-row and reported in the result list, exactly as in
     * the non-replay path.
     */
    fun bulk(adminId: UUID, idempotencyKey: String, req: BulkPayoutsRequest): BulkPayoutResultDto {
        // Fast path: a prior bulk run with this (admin_id, idempotency_key)
        // already minted a group_id — reuse it so the dashboard kpi
        // (`COUNT(DISTINCT group_id)`) still collapses retries into one
        // logical decision and the FE bulk-undo button has a stable id.
        val groupId = adminDecisionLogRepository.findGroupIdByAdminAndIdempotencyKey(
            adminId,
            idempotencyKey,
        ) ?: UUID.randomUUID()

        val results = try {
            req.ids.map { id ->
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
        if (cycle.status == PayoutCycleStatus.PAID) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.INVALID_STATE_TRANSITION, "Already paid")
        }
        cycle.holdReason = null
        cycle.status = PayoutCycleStatus.PENDING
        cycle.settledAt = cycle.settledAt ?: OffsetDateTime.now()
        payoutCycleRepository.save(cycle)

        // C-3: per-row dedup gate. On a retry, an audit-log row already
        // exists for (admin, key, payoutId) — skip the second
        // logPayoutDecision + pushMessage so the photographer inbox
        // doesn't get a duplicate "approved" notification and the audit
        // trail stays a single logical decision.
        if (adminDecisionLogRepository.existsByAdminIdAndIdempotencyKeyAndTargetPayoutId(
                adminId,
                idempotencyKey,
                payoutId,
            )
        ) {
            return
        }

        val decision = adminDecisionLogService.logPayoutDecision(
            adminId = adminId,
            targetPayoutId = cycle.id,
            decision = "approved",
            groupId = groupId,
            idempotencyKey = idempotencyKey,
        )
        adminDecisionLogService.pushMessage(
            photographerId = cycle.photographerId,
            kind = PhotographerMessageKind.PAYOUT_APPROVED,
            body = "Payout cycle ${cycle.id} was approved.",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
    }

    private fun bulkHold(
        adminId: UUID,
        payoutId: String,
        reason: String,
        groupId: UUID,
        idempotencyKey: String,
    ) {
        val cycle = loadCycle(payoutId)
        if (cycle.status == PayoutCycleStatus.PAID) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.INVALID_STATE_TRANSITION, "Already paid")
        }
        cycle.status = PayoutCycleStatus.HELD
        cycle.holdReason = reason.take(255)
        payoutCycleRepository.save(cycle)

        // C-3: per-row dedup gate, see bulkApprove above.
        if (adminDecisionLogRepository.existsByAdminIdAndIdempotencyKeyAndTargetPayoutId(
                adminId,
                idempotencyKey,
                payoutId,
            )
        ) {
            return
        }

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
     * approved=PENDING+settledAt) are skipped so the generator cannot clobber
     * human judgment.
     *
     * Eligibility: any photographer with net amount > ₱0 in the window. The
     * cycle's payout method is the photographer's primary account method, or
     * GCASH as a safe default when no account is set — the admin still sees
     * an empty payoutAccount snapshot in the queue and can hold for setup.
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

        for (row in rows) {
            val photographerId = row[0] as UUID
            val amount = row[1] as BigDecimal
            val itemCount = (row[2] as Number).toInt()

            val settings = photographerSettingsRepository.findById(photographerId).orElse(null)
            val handle = settings?.handle
            val cycleId = buildCycleId(weekOf, handle)

            val primaryAccount = payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId)
            val method = primaryAccount?.method ?: PayoutMethod.GCASH

            val existing = payoutCycleRepository.findById(cycleId).orElse(null)

            when {
                existing == null -> {
                    val cycle = PayoutCycle(
                        id = cycleId,
                        photographerId = photographerId,
                        weekOf = weekOf,
                        methodWire = method.wire,
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
                existing.status == PayoutCycleStatus.PAID ||
                    existing.status == PayoutCycleStatus.HELD ||
                    (existing.status == PayoutCycleStatus.PENDING && existing.settledAt != null) -> {
                    // Paid, held, or already-approved (PENDING + settledAt set) —
                    // leave alone. Re-summing would silently rewrite an amount
                    // the admin already decided on.
                    skipped++
                }
                else -> {
                    existing.amountPhp = amount
                    existing.itemCount = itemCount
                    existing.method = method
                    existing.status = PayoutCycleStatus.PENDING
                    existing.settledAt = null
                    payoutCycleRepository.save(existing)
                    updated++
                    totalAmount = totalAmount.add(amount)
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
        payoutCycleRepository.findById(payoutId).orElseThrow {
            NotFoundException(code = ErrorCodes.PAYOUT_NOT_FOUND, message = "Payout cycle not found")
        }

    // FE filter values pending_review / approved / held / paid map to our DB
    // states differently — pending_review and approved both live as DB
    // PENDING. For list filtering we accept either FE-form value: the
    // listing query just constrains on the DB enum value. approved is a
    // presentation-only state; the FE relies on settledAt being non-null
    // to differentiate it from pending_review.
    private fun mapAdminStatusFilter(filter: String): String? {
        return when (filter.trim().lowercase()) {
            "pending_review", "pending" -> PayoutCycleStatus.PENDING.wire
            "approved" -> PayoutCycleStatus.PENDING.wire
            "held" -> PayoutCycleStatus.HELD.wire
            "paid" -> PayoutCycleStatus.PAID.wire
            "scheduled" -> PayoutCycleStatus.SCHEDULED.wire
            else -> null
        }
    }

    private fun hydrateMany(cycles: List<PayoutCycle>): List<AdminPayoutCycleDto> {
        if (cycles.isEmpty()) return emptyList()
        return cycles.map { hydrateOne(it) }
    }

    internal fun hydrateOne(cycle: PayoutCycle): AdminPayoutCycleDto {
        val user = userRepository.findById(cycle.photographerId).orElse(null)
        val settings = photographerSettingsRepository.findById(cycle.photographerId).orElse(null)
        val account = payoutAccountRepository.findByUserIdAndIsPrimaryTrue(cycle.photographerId)
            ?: payoutAccountRepository.findAllByUserIdOrderByCreatedAtAsc(cycle.photographerId).firstOrNull()
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
            reviewedAt = if (cycle.status == PayoutCycleStatus.PENDING && cycle.settledAt != null)
                cycle.settledAt?.toString()
            else null,
            paidAt = if (cycle.status == PayoutCycleStatus.PAID) cycle.settledAt?.toString() else null,
            paymentReference = cycle.paymentReference,
            holdReason = cycle.holdReason,
            payoutAccount = snapshotAccount(account),
        )
    }

    private fun toAdminWireStatus(cycle: PayoutCycle): String {
        return when (cycle.status) {
            // approved is settledAt-set + still PENDING; before that it's
            // pending_review for the admin queue.
            PayoutCycleStatus.PENDING -> if (cycle.settledAt != null) "approved" else "pending_review"
            PayoutCycleStatus.SCHEDULED -> "pending_review"
            PayoutCycleStatus.HELD -> "held"
            PayoutCycleStatus.PAID -> "paid"
        }
    }

    private fun snapshotAccount(account: PayoutAccount?): AdminPayoutAccountSnapshotDto {
        if (account == null) {
            return AdminPayoutAccountSnapshotDto(
                method = "",
                accountNumber = "",
                accountName = "",
                qr = null,
            )
        }
        val qr = account.qrS3Key?.let { key ->
            AdminPayoutQrDto(
                // Reuse the cover TTL — both are private images surfaced via
                // a presigned GET that the FE renders straight into <img src>.
                dataUrl = storageService.presignedGetUrl(key, storageProperties.presignedTtl.cover),
                uploadedAt = account.createdAt.toString(),
            )
        }
        return AdminPayoutAccountSnapshotDto(
            method = account.method.wire,
            accountNumber = account.accountNumber,
            accountName = account.accountName,
            qr = qr,
        )
    }
}
