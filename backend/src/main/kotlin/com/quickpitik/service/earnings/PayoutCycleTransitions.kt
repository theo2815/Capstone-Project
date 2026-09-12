package com.quickpitik.service.earnings

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.PayoutCycleRepository
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Component
import java.util.UUID

/**
 * Every legal move in the payout lifecycle, and the single writer of
 * `payout_cycles.status`.
 *
 * Before this, the table was spelled out at each call site — admin with a
 * per-method `ensureFrom(setOf(...))`, the transfer webhook with an inline
 * `when`, the photographer path with an inline `if` — and two had already
 * drifted: `hold()` refused a HELD cycle while `bulkHold()` accepted one and
 * wrote a HELD -> HELD row into `payout_status_history` (audit 2026-09-11,
 * items 1 and 3). Sharing one table is what makes that drift unrepresentable.
 *
 * Sits beside [PayoutStatusRecorder], which is the same pattern one level down:
 * that class is the single writer of the audit trail, this one is the single
 * writer of the status it records — and [apply] writes both together, so a
 * status change can no longer forget its history row.
 *
 * Creation is deliberately NOT here. A cycle is born from
 * `PayoutRequestService.request()` (photographer) or
 * `AdminPayoutService.generateForWeek` (weekly backfill), and neither is a
 * lifecycle move — see `PayoutCycleService`'s KDoc for why both paths exist.
 */
@Component
class PayoutCycleTransitions(
    private val payoutCycleRepository: PayoutCycleRepository,
    private val payoutStatusRecorder: PayoutStatusRecorder,
) {
    /**
     * The guard on its own, for a caller that must reject an illegal move
     * *before* it does something irreversible — `markProcessing` checks this
     * before posting the PayMongo transfer, so a refused transition can never
     * follow money that has already left.
     */
    fun ensureAllowed(cycle: PayoutCycle, transition: PayoutTransition) {
        if (cycle.status == transition.to) return
        if (cycle.status !in transition.allowedFrom) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Cannot do that to a ${cycle.statusWire} payout",
            )
        }
    }

    /**
     * Guard, move, persist and record as one step.
     *
     * Returns `false` when the cycle already holds the target state: the
     * caller's other field edits are still saved, but no second history row is
     * written. A double-clicked hold is a no-op — not a 409, and not a
     * HELD -> HELD row — which matches the already-applied skips that
     * `PayoutRequestService.withdraw` and the transfer webhook already use.
     */
    fun apply(
        cycle: PayoutCycle,
        transition: PayoutTransition,
        actorId: UUID?,
        reason: String? = null,
    ): Boolean {
        val from = cycle.status
        ensureAllowed(cycle, transition)
        if (from == transition.to) {
            payoutCycleRepository.save(cycle)
            return false
        }
        cycle.status = transition.to
        payoutCycleRepository.save(cycle)
        payoutStatusRecorder.record(cycle.id, from, transition.to, transition.actorType, actorId, reason)
        return true
    }
}

/**
 * Named per actor, not per target state, because admin and the provider hold
 * genuinely different authority over the same destination: admin may pay a
 * cycle that is merely APPROVED, the webhook may only confirm one already
 * PROCESSING; the webhook may return a PROCESSING transfer, admin may only
 * return a PAID one. A table keyed by target alone would quietly widen both.
 */
enum class PayoutTransition(
    val to: PayoutCycleStatus,
    val allowedFrom: Set<PayoutCycleStatus>,
    val actorType: String,
) {
    AdminApprove(PayoutCycleStatus.APPROVED, setOf(PayoutCycleStatus.PENDING, PayoutCycleStatus.HELD), "admin"),
    AdminHold(
        PayoutCycleStatus.HELD,
        setOf(PayoutCycleStatus.PENDING, PayoutCycleStatus.APPROVED, PayoutCycleStatus.PROCESSING),
        "admin",
    ),
    AdminReject(
        PayoutCycleStatus.REJECTED,
        setOf(PayoutCycleStatus.PENDING, PayoutCycleStatus.HELD, PayoutCycleStatus.APPROVED),
        "admin",
    ),
    AdminMarkProcessing(PayoutCycleStatus.PROCESSING, setOf(PayoutCycleStatus.APPROVED), "admin"),
    AdminMarkPaid(
        PayoutCycleStatus.PAID,
        setOf(PayoutCycleStatus.APPROVED, PayoutCycleStatus.PROCESSING),
        "admin",
    ),
    AdminMarkFailed(PayoutCycleStatus.FAILED, setOf(PayoutCycleStatus.PROCESSING), "admin"),
    AdminMarkReturned(PayoutCycleStatus.RETURNED, setOf(PayoutCycleStatus.PAID), "admin"),
    PhotographerCancel(PayoutCycleStatus.CANCELLED, setOf(PayoutCycleStatus.HELD), "photographer"),
    ProviderPaid(PayoutCycleStatus.PAID, setOf(PayoutCycleStatus.PROCESSING), "provider"),
    ProviderFailed(PayoutCycleStatus.FAILED, setOf(PayoutCycleStatus.PROCESSING), "provider"),
    ProviderReturned(
        PayoutCycleStatus.RETURNED,
        setOf(PayoutCycleStatus.PROCESSING, PayoutCycleStatus.PAID),
        "provider",
    ),
}
