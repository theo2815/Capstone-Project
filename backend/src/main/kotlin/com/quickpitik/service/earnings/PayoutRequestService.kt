package com.quickpitik.service.earnings

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.earnings.PayoutBalanceDto
import com.quickpitik.dto.earnings.PhotographerPayoutDto
import com.quickpitik.dto.earnings.buildSequencedCycleId
import com.quickpitik.dto.earnings.toDto
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.TransactionRepository
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
 *   - One open request at a time (status IN pending / held).
 *   - Minimum unpaid balance: ₱500 (lifetime kept − paid cycles).
 *   - Primary payout account must exist.
 *   - Cycle IDs are sequence-based per photographer to avoid the week-based
 *     collision when a photographer requests twice in the same ISO week.
 *
 * Withdraw is allowed only when the cycle is on hold — the held state is
 * specifically the signal for the photographer to act (e.g. update payout
 * account). Hard-delete on withdraw frees the photographer to re-request
 * once the underlying issue is fixed; the audit log row (if any) persists.
 */
@Service
@Transactional
class PayoutRequestService(
    private val payoutCycleRepository: PayoutCycleRepository,
    private val transactionRepository: TransactionRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val payoutAccountRepository: PayoutAccountRepository,
) {

    @Transactional(readOnly = true)
    fun getBalance(photographerId: UUID): PayoutBalanceDto {
        val balance = computeUnpaidBalance(photographerId)
        val open = payoutCycleRepository.findFirstByPhotographerIdAndStatusWireIn(
            photographerId,
            OPEN_STATUS_WIRES,
        )
        return PayoutBalanceDto(
            unpaidBalance = balance,
            minimum = MINIMUM,
            hasOpenRequest = open != null,
            openRequest = open?.toDto(),
        )
    }

    fun request(photographerId: UUID): PhotographerPayoutDto {
        // One open at a time. Pending review, approved (PENDING+settledAt),
        // and held all count as "open" — admin must resolve the prior cycle
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

        val balance = computeUnpaidBalance(photographerId)
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

        return payoutCycleRepository.save(cycle).toDto()
    }

    fun withdraw(photographerId: UUID, cycleId: String) {
        val cycle = payoutCycleRepository.findByIdAndPhotographerId(cycleId, photographerId)
            ?: throw NotFoundException(
                code = ErrorCodes.PAYOUT_NOT_FOUND,
                message = "Payout request not found",
            )
        if (cycle.status != PayoutCycleStatus.HELD) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Can only withdraw a request that is on hold",
            )
        }
        payoutCycleRepository.delete(cycle)
    }

    private fun computeUnpaidBalance(photographerId: UUID): BigDecimal {
        val lifetime = transactionRepository.sumLifetime(photographerId)
        val paid = payoutCycleRepository.sumPaidByPhotographer(photographerId)
        return lifetime.subtract(paid)
    }

    private companion object {
        val MINIMUM: BigDecimal = BigDecimal("500")
        val OPEN_STATUS_WIRES = listOf("pending", "held")
        val EPOCH: OffsetDateTime = OffsetDateTime.parse("1970-01-01T00:00:00Z")
    }
}
