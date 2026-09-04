package com.quickpitik.service.earnings

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.earnings.PhotographerPayoutDto
import com.quickpitik.dto.earnings.toDto
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.repository.PayoutCycleRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

/**
 * Read-side for payout cycles.
 *
 * Cycles are created by `PayoutRequestService` (photographer-initiated, since
 * 2026-05-19) or by `AdminPayoutService.generateForWeek` as an optional
 * backfill. A `seed()` helper used to live here for the PR 9 smoke harness; it
 * was removed 2026-08-16 with zero callers, having been superseded by both of
 * those paths. The website's old "add a guard to seed()" request is answered by
 * that deletion — there is no longer a path to guard.
 */
@Service
@Transactional
class PayoutCycleService(
    private val payoutCycleRepository: PayoutCycleRepository,
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

}
