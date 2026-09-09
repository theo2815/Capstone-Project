package com.quickpitik.service.earnings

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.earnings.PayoutBalanceAccountDto
import com.quickpitik.dto.earnings.PayoutBalanceDto
import com.quickpitik.dto.earnings.PhotographerPayoutDto
import com.quickpitik.dto.earnings.buildSequencedCycleId
import com.quickpitik.dto.earnings.toDto
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutAccountAuditRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.time.DayOfWeek
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.UUID

/**
 * Photographer-initiated payout requests. Replaces the prior weekly-cron model
 * (admin-generated cycles). A photographer with ≥ ₱500 unpaid balance and no
 * existing open request can submit a request; admin reviews and processes
 * through the existing AdminPayoutService approve/hold/mark-paid pipeline.
 *
 * Constraints:
 *   - One open request at a time (pending / approved / processing / held).
 *   - Configured minimum available balance (cleared earnings minus paid and reserved).
 *   - Primary payout account must have a QR and be past its change cooldown.
 *   - Cycle IDs are sequence-based per photographer to avoid the week-based
 *     collision when a photographer requests twice in the same ISO week.
 *
 * Withdraw is allowed only when the cycle is on hold — the held state is
 * specifically the signal for the photographer to act (e.g. update payout
 * account). Withdraw SOFT-cancels (status → cancelled), preserving the row for
 * the audit trail while freeing the photographer to request again.
 */
@Service
@Transactional
class PayoutRequestService(
    private val payoutCycleRepository: PayoutCycleRepository,
    private val transactionRepository: TransactionRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val payoutAccountRepository: PayoutAccountRepository,
    private val payoutAccountAuditRepository: PayoutAccountAuditRepository,
    private val payoutStatusRecorder: PayoutStatusRecorder,
    private val platformProperties: PlatformProperties,
    private val userRepository: UserRepository,
) {

    @Transactional(readOnly = true)
    fun getBalance(photographerId: UUID): PayoutBalanceDto {
        val balances = computeBalances(photographerId)
        val open = payoutCycleRepository.findFirstByPhotographerIdAndStatusWireIn(
            photographerId,
            OPEN_STATUS_WIRES,
        )
        // Same lookup request() uses to enforce PAYOUT_NO_ACCOUNT — surfacing it
        // here lets the FE gate on the authoritative DB state, not its store.
        val primary = payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId)
        // Surface the post-account-change cooldown so the FE can explain the hold
        // instead of letting the photographer hit request() and get a 409.
        val cooldownUntil = payoutAccountAuditRepository
            .findFirstByUserIdOrderByCreatedAtDesc(photographerId)
            ?.createdAt?.plus(platformProperties.payoutAccountCooldown)
            ?.takeIf { it.isAfter(OffsetDateTime.now()) }
        return PayoutBalanceDto(
            unpaidBalance = balances.available,
            clearing = balances.clearing,
            reserved = balances.reserved,
            accountCooldownUntil = cooldownUntil,
            minimum = MINIMUM,
            hasOpenRequest = open != null,
            openRequest = open?.toDto(),
            primaryAccount = primary?.let {
                PayoutBalanceAccountDto(
                    method = it.method.wire,
                    accountNumber = it.accountNumber,
                    accountName = it.accountName,
                    hasQr = !it.qrS3Key.isNullOrBlank(),
                )
            },
        )
    }

    fun request(photographerId: UUID): PhotographerPayoutDto {
        // Shares the destination mutation lock so cooldown + QR snapshot are
        // checked against one committed account version, including first setup.
        userRepository.lockById(photographerId)
        // One open at a time. Pending, approved, processing, and held all
        // count as "open" — admin must resolve the prior cycle
        // before the photographer can submit another.
        payoutCycleRepository.findFirstByPhotographerIdAndStatusWireIn(
            photographerId,
            OPEN_STATUS_WIRES,
        )?.let {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.PAYOUT_REQUEST_OPEN,
                message = "You already have an open payout request",
            )
        }

        // Anti-fraud cooldown: a recent payout-account change bars requests, so a
        // compromised account can't change the destination and withdraw at once.
        val cooldownStart = OffsetDateTime.now().minus(platformProperties.payoutAccountCooldown)
        if (payoutAccountAuditRepository.existsByUserIdAndCreatedAtAfter(photographerId, cooldownStart)) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.PAYOUT_ACCOUNT_COOLDOWN,
                message = "Payout is on a short hold after a recent payout-account change. Try again later.",
            )
        }

        val balance = computeBalances(photographerId).available
        if (balance < MINIMUM) {
            throw ValidationException(
                code = ErrorCodes.PAYOUT_BELOW_MINIMUM,
                message = "Unpaid balance is below the ₱$MINIMUM minimum",
                field = "balance",
            )
        }

        val primary = payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId)
            ?: throw ValidationException(
                code = ErrorCodes.PAYOUT_NO_ACCOUNT,
                message = "Set a primary payout account before requesting a payout",
                field = "payoutAccount",
            )

        if (primary.qrS3Key.isNullOrBlank()) {
            throw ValidationException(
                code = ErrorCodes.PAYOUT_QR_REQUIRED,
                message = "Upload a payment QR code for your primary payout account before requesting a payout",
                field = "payoutAccount",
            )
        }

        val settings = photographerSettingsRepository.findById(photographerId).orElse(null)
        val handle = settings?.handle
        val sequence = payoutCycleRepository.countByPhotographerId(photographerId) + 1
        val cycleId = buildSequencedCycleId(handle, photographerId, sequence)

        // weekOf stays populated with the request date's ISO Monday so the
        // FE's existing "cycle of {date}" copy still reads sensibly.
        val today = LocalDate.now(ZoneId.of("Asia/Manila"))
        val weekOf = today.minusDays((today.dayOfWeek.value - DayOfWeek.MONDAY.value).toLong())

        // itemCount = count of non-refund sales over the photographer's whole
        // history. Informational — the money flow keys off amountPhp. A finer
        // "since last paid cycle" count would need a max(settledAt) join and
        // isn't justified at capstone scope.
        val itemCount = transactionRepository.countSalesWindow(
            photographerId = photographerId,
            from = EPOCH,
            to = OffsetDateTime.now(),
        ).toInt()

        val cycle = PayoutCycle(
            id = cycleId,
            photographerId = photographerId,
            weekOf = weekOf,
            methodWire = primary.method.wire,
        )
        cycle.amountPhp = balance
        cycle.itemCount = itemCount
        cycle.method = primary.method
        cycle.status = PayoutCycleStatus.PENDING
        cycle.settledAt = null
        // Snapshot the destination now (immutable) so a later payout-account
        // change can't reroute this in-flight payout, and disbursement targets a
        // fixed account (V58, Phase 8).
        cycle.accountSnapshotMethod = primary.method.wire
        cycle.accountSnapshotNumber = primary.accountNumber
        cycle.accountSnapshotName = primary.accountName
        cycle.accountSnapshotQrKey = primary.qrS3Key

        val saved = try {
            payoutCycleRepository.saveAndFlush(cycle)
        } catch (ex: DataIntegrityViolationException) {
            // Lost the race: the partial unique index uq_payout_open_per_photographer
            // rejected a second concurrent open cycle. Same outcome as the findFirst
            // pre-check above, so surface the same clean 409 (not a raw 500).
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.PAYOUT_REQUEST_OPEN,
                message = "You already have an open payout request",
            )
        }
        payoutStatusRecorder.record(
            cycleId = saved.id,
            from = null,
            to = PayoutCycleStatus.PENDING,
            actorType = "photographer",
            actorId = photographerId,
        )
        return saved.toDto()
    }

    fun withdraw(photographerId: UUID, cycleId: String) {
        // Ownership gated by the (id, photographerId) lookup.
        val cycle = payoutCycleRepository.lockByIdAndPhotographerId(cycleId, photographerId)
            ?: return // gone entirely → idempotent no-op (client retry after success)
        // Already cancelled → idempotent no-op (double-click / retry after success).
        if (cycle.status == PayoutCycleStatus.CANCELLED) return
        // A non-HELD cycle is a genuine state error, not a replay — keep the 409.
        if (cycle.status != PayoutCycleStatus.HELD || cycle.provider != "manual") {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Can only cancel a manual payout request that is on hold",
            )
        }
        // Soft-cancel: keep the row for the audit trail (was a hard delete, which
        // destroyed the history). CANCELLED is terminal + not "open", so it releases
        // the reservation and frees the photographer to request again.
        cycle.status = PayoutCycleStatus.CANCELLED
        payoutCycleRepository.save(cycle)
        payoutStatusRecorder.record(
            cycleId = cycle.id,
            from = PayoutCycleStatus.HELD,
            to = PayoutCycleStatus.CANCELLED,
            actorType = "photographer",
            actorId = photographerId,
        )
    }

    private data class Balances(
        val available: BigDecimal,
        val clearing: BigDecimal,
        val reserved: BigDecimal,
    )

    /**
     * Withdrawable ("available") balance and its two withheld parts, all derived
     * from the ledger — nothing is stored. `available = max(0, cleared − paid −
     * reserved)`:
     *   - cleared: sales aged past the clearing window PLUS all refunds (refunds
     *     bite immediately). Money younger than the window is not withdrawable.
     *   - paid: already-settled cycles.
     *   - reserved: amounts tied up in open cycles (pending/held).
     * `clearing` is the not-yet-withdrawable remainder (lifetime − cleared). The
     * non-negative floor stops a refund-after-payout from showing (or paying) a
     * negative balance.
     *
     * NOTE: read here WITHOUT a lock — Phase 2 adds the pessimistic lock + the
     * partial unique index that make concurrent requests safe. This method is the
     * amount source of truth; the client never supplies the amount.
     */
    private fun computeBalances(photographerId: UUID): Balances {
        val cutoff = OffsetDateTime.now().minus(platformProperties.payoutClearingWindow)
        val cleared = transactionRepository.sumCleared(photographerId, cutoff)
        val lifetime = transactionRepository.sumLifetime(photographerId)
        val paid = payoutCycleRepository.sumPaidByPhotographer(photographerId)
        val reserved = payoutCycleRepository.sumReservedByPhotographer(photographerId)
        val available = cleared.subtract(paid).subtract(reserved).max(BigDecimal.ZERO)
        val clearing = lifetime.subtract(cleared).max(BigDecimal.ZERO)
        return Balances(available = available, clearing = clearing, reserved = reserved)
    }

    private companion object {
        val MINIMUM: BigDecimal = BigDecimal("500")
        // MUST match the V53 uq_payout_open_per_photographer index AND
        // sumReservedByPhotographer — all three define "open" (funds reserved,
        // slot locked). If these drift, an approved/processing cycle stops
        // reading as an open request: the hero shows "Available to request" and
        // the next request() 409s on the index instead of the clean pre-check.
        val OPEN_STATUS_WIRES = listOf("scheduled", "pending", "approved", "processing", "held")
        val EPOCH: OffsetDateTime = OffsetDateTime.parse("1970-01-01T00:00:00Z")
    }
}
