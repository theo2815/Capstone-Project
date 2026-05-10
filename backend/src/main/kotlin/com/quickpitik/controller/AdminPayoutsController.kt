package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminPayoutCycleDto
import com.quickpitik.dto.admin.BulkPayoutResultDto
import com.quickpitik.dto.admin.BulkPayoutsRequest
import com.quickpitik.dto.admin.HoldPayoutRequest
import com.quickpitik.dto.admin.MarkPayoutPaidRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminPayoutService
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

@RestController
@RequestMapping("/api/v1/admin/payouts")
@PreAuthorize("hasRole('ADMIN')")
class AdminPayoutsController(
    private val adminPayoutService: AdminPayoutService,
) {

    @GetMapping
    fun list(
        @RequestParam(required = false) status: String?,
        @RequestParam(required = false) q: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<AdminPayoutCycleDto> =
        adminPayoutService.list(status, q, PaginationParams.of(offset, limit))

    @PostMapping("/{payoutId}/approve")
    fun approve(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable payoutId: String,
    ): AdminPayoutCycleDto = adminPayoutService.approve(principal.userId, payoutId)

    @PostMapping("/{payoutId}/hold")
    fun hold(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable payoutId: String,
        @Valid @RequestBody body: HoldPayoutRequest,
    ): AdminPayoutCycleDto = adminPayoutService.hold(principal.userId, payoutId, body.reason)

    @PostMapping("/{payoutId}/mark-paid")
    fun markPaid(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable payoutId: String,
        @Valid @RequestBody body: MarkPayoutPaidRequest,
    ): AdminPayoutCycleDto = adminPayoutService.markPaid(principal.userId, payoutId, body)

    @PostMapping("/bulk")
    fun bulk(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: BulkPayoutsRequest,
    ): BulkPayoutResultDto = adminPayoutService.bulk(principal.userId, body)
}
