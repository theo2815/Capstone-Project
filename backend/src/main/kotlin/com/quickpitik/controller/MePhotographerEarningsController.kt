package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.earnings.EarningsOverviewDto
import com.quickpitik.dto.earnings.PayoutBalanceDto
import com.quickpitik.dto.earnings.PayoutReportDto
import com.quickpitik.dto.earnings.PhotographerPayoutDto
import com.quickpitik.dto.earnings.PhotographerPhotoPurchasesResponse
import com.quickpitik.dto.earnings.PhotographerPurchasedPhotoDto
import com.quickpitik.dto.earnings.SubmitPayoutReportRequest
import com.quickpitik.dto.earnings.TransactionsLedgerResponse
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.earnings.EarningsService
import com.quickpitik.service.earnings.PayoutReportService
import com.quickpitik.service.earnings.PayoutRequestService
import com.quickpitik.service.earnings.PhotographerPurchaseService
import com.quickpitik.service.earnings.TransactionsLedgerService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

/**
 * Phase F.3 photographer-earnings + billing read surface. Photographer-scoped
 * via class-level @PreAuthorize. The /payouts cycle branch lives in PR 8's
 * MePhotographerSettingsController per the path-overlap dispatch (Q-009) — it
 * was wired here to a real PayoutCycleService call as part of this PR.
 *
 * The report submission endpoint accepts a string cycle id (e.g. `PAY-{HANDLE}-
 * {NUMBER}`, see `buildSequencedCycleId`) as the path variable, which is distinct from the UUID-typed
 * payout-account id PR 8 uses for its CRUD. Spring routes correctly because
 * `/payouts/{id}/report` is more specific than `/payouts/{id}`.
 */
@RestController
@RequestMapping("/api/v1/me/photographer")
@PreAuthorize("hasRole('PHOTOGRAPHER')")
class MePhotographerEarningsController(
    private val earningsService: EarningsService,
    private val payoutReportService: PayoutReportService,
    private val transactionsLedgerService: TransactionsLedgerService,
    private val payoutRequestService: PayoutRequestService,
    private val photographerPurchaseService: PhotographerPurchaseService,
    private val rateLimiter: RateLimiter,
) {
    @GetMapping("/events/{eventId}/purchased-photos")
    fun listPurchasedPhotos(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<PhotographerPurchasedPhotoDto> {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_ORDER_READ, principal.userId.toString())
        return photographerPurchaseService.listPurchasedPhotos(principal.userId, eventId, PaginationParams.of(offset, limit))
    }

    @GetMapping("/photos/{photoId}/purchases")
    fun listPhotoPurchases(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable photoId: UUID,
        @RequestParam(required = false) orderId: UUID?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PhotographerPhotoPurchasesResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_ORDER_READ, principal.userId.toString())
        return photographerPurchaseService.listPhotoPurchases(principal.userId, photoId, orderId, PaginationParams.of(offset, limit))
    }

    @GetMapping("/earnings")
    fun getEarnings(@AuthenticationPrincipal principal: AuthPrincipal): EarningsOverviewDto =
        earningsService.getOverview(principal.userId)

    @GetMapping("/earnings/per-event")
    fun listPerEventEarnings(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestParam(required = false) offset: Int?,
        // FE default is 8 — keep it aligned so the contract is uniform whether
        // the FE specifies a limit or not. PaginationParams clamps to MAX_LIMIT.
        @RequestParam(required = false) limit: Int?,
    ) = earningsService.listPerEvent(
        photographerId = principal.userId,
        params = PaginationParams.of(offset, limit ?: PER_EVENT_DEFAULT_LIMIT),
    )

    @PostMapping("/payouts/{cycleId}/report")
    fun submitPayoutReport(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable cycleId: String,
        @Valid @RequestBody body: SubmitPayoutReportRequest,
    ): PayoutReportDto = payoutReportService.submit(
        photographerId = principal.userId,
        cycleId = cycleId,
        request = body,
    )

    @GetMapping("/payouts/reports")
    fun listPayoutReports(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestParam(required = false) cycleId: String?,
        @RequestParam(required = false) status: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<PayoutReportDto> = payoutReportService.list(
        photographerId = principal.userId,
        cycleId = cycleId?.takeIf { it.isNotBlank() },
        statusWire = status?.takeIf { it.isNotBlank() },
        params = PaginationParams.of(offset, limit),
    )

    // ─── Photographer-initiated payout requests ────────────────────────
    // /payouts/balance — drives the Request-payout hero state machine on
    // /dashboard/billing. Returns the unpaid balance, minimum threshold, and
    // the photographer's current open request (if any).
    @GetMapping("/payouts/balance")
    fun getPayoutBalance(
        @AuthenticationPrincipal principal: AuthPrincipal,
    ): PayoutBalanceDto = payoutRequestService.getBalance(principal.userId)

    // /payouts/request — the photographer initiates a payout. Validates the
    // ₱500 minimum, one-open-at-a-time, and primary-account-set rules. Creates
    // a PayoutCycle in pending_review; admin processes it through the existing
    // approve/hold/mark-paid pipeline.
    @PostMapping("/payouts/request")
    fun requestPayout(
        @AuthenticationPrincipal principal: AuthPrincipal,
    ): PhotographerPayoutDto = payoutRequestService.request(principal.userId)

    // /payouts/{id}/withdraw — photographer cancels a HELD request after
    // fixing whatever the admin flagged. Soft-cancels the cycle (the row stays
    // for the audit trail) so the photographer can submit a new request
    // immediately. The path doesn't
    // collide with /payouts/{cycleId}/report — different terminal segments.
    @PostMapping("/payouts/{id}/withdraw")
    fun withdrawPayout(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: String,
    ) {
        payoutRequestService.withdraw(principal.userId, id)
    }

    @GetMapping("/billing/transactions")
    fun listTransactions(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestParam(required = false) offset: Int?,
        // FE default is 25 (sticky-header sized). Page math + monthTotals are
        // both computed server-side.
        @RequestParam(required = false) limit: Int?,
    ): TransactionsLedgerResponse = transactionsLedgerService.list(
        photographerId = principal.userId,
        params = PaginationParams.of(offset, limit ?: BILLING_DEFAULT_LIMIT),
    )

    private companion object {
        const val PER_EVENT_DEFAULT_LIMIT = 8
        const val BILLING_DEFAULT_LIMIT = 25
    }
}
