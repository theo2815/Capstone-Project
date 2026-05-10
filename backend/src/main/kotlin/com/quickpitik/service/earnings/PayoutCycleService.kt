package com.quickpitik.service.earnings

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.earnings.CreatePayoutCycleRequest
import com.quickpitik.dto.earnings.PhotographerPayoutDto
import com.quickpitik.dto.earnings.buildCycleId
import com.quickpitik.dto.earnings.toDto
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.PayoutMethod
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

/**
 * Read-side for payout cycles. Cycle generation (the weekly cron / admin
 * trigger that creates rows from cumulative transactions) is icebox for PR 9 —
 * the seed helper here exists so the smoke test can mint a cycle by hand and
 * exercise the read paths end-to-end. Admin (PR 10) replaces it with a real
 * generator.
 */
@Service
@Transactional
class PayoutCycleService(
    private val payoutCycleRepository: PayoutCycleRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
) {
    @Transactional(readOnly = true)
    fun list(
        photographerId: UUID,
        params: PaginationParams,
    ): PaginatedResponse<PhotographerPayoutDto> {
        val page = payoutCycleRepository.pageForPhotographer(
            photographerId = photographerId,
            pageable = OffsetLimitPageable(params),
        )
        if (page.isEmpty) return PaginatedResponse.empty(params)
        val items = page.content.map { it.toDto() }
        return PaginatedResponse.of(items, page.totalElements, params)
    }

    @Transactional(readOnly = true)
    fun findOwned(photographerId: UUID, cycleId: String): PayoutCycle? =
        payoutCycleRepository.findByIdAndPhotographerId(cycleId, photographerId)

    /**
     * Test-seed helper used by the smoke harness + admin (PR 10). Idempotent
     * on (photographer, week_of) per the unique index — re-seeding the same
     * week updates the existing row in place rather than throwing. The cycle
     * id is rebuilt from the photographer's current handle each call, so the
     * id stays stable as long as the handle stays stable.
     */
    fun seed(photographerId: UUID, request: CreatePayoutCycleRequest): PhotographerPayoutDto {
        val method = PayoutMethod.fromWire(request.method) ?: throw ValidationException(
            code = "VALIDATION_ERROR",
            message = "method must be one of gcash|maya|gotyme",
            field = "method",
        )
        val statusEnum = request.status?.let { PayoutCycleStatus.fromWire(it) }
            ?: PayoutCycleStatus.SCHEDULED
        val handle = photographerSettingsRepository.findById(photographerId)
            .map { it.handle }
            .orElse(null)
        val cycleId = buildCycleId(request.weekOf, handle)

        val cycle = payoutCycleRepository.findById(cycleId).orElseGet {
            PayoutCycle(
                id = cycleId,
                photographerId = photographerId,
                weekOf = request.weekOf,
                methodWire = method.wire,
            )
        }
        cycle.weekOf = request.weekOf
        cycle.amountPhp = request.amountPhp
        cycle.itemCount = request.itemCount
        cycle.method = method
        cycle.status = statusEnum
        cycle.settledAt = request.settledAt
        cycle.paymentReference = request.paymentReference
        return payoutCycleRepository.save(cycle).toDto()
    }
}
