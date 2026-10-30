package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminDisputeDto
import com.quickpitik.dto.admin.DenyDisputeRequest
import com.quickpitik.dto.admin.EscalateDisputeRequest
import com.quickpitik.dto.admin.ResolveDisputeRequest
import com.quickpitik.dto.admin.SendRefundLinkRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminDisputeService
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

@RestController
@RequestMapping("/api/v1/admin/disputes")
@PreAuthorize("hasRole('ADMIN')")
class AdminDisputesController(
    private val adminDisputeService: AdminDisputeService,
    private val rateLimiter: RateLimiter,
) {

    @GetMapping
    fun list(
        @RequestParam(required = false) status: String?,
        @RequestParam(required = false) q: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<AdminDisputeDto> {
        @Suppress("UNUSED_VARIABLE") val _q = q
        return adminDisputeService.list(status, PaginationParams.of(offset, limit))
    }

    @PostMapping("/{disputeId}/resolve")
    fun resolve(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable disputeId: UUID,
        @Valid @RequestBody body: ResolveDisputeRequest,
    ): AdminDisputeDto = adminDisputeService.resolve(principal.userId, disputeId, body)

    @PostMapping("/{disputeId}/refund-link")
    fun sendRefundLink(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable disputeId: UUID,
        @Valid @RequestBody body: SendRefundLinkRequest,
    ): AdminDisputeDto {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, principal.userId.toString())
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_REFUND_LINK_PER_DISPUTE, "dispute:$disputeId")
        return adminDisputeService.sendRefundLink(principal.userId, disputeId, body.url)
    }

    @PostMapping("/{disputeId}/mark-refunded")
    fun markRefunded(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable disputeId: UUID,
    ): AdminDisputeDto {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, principal.userId.toString())
        return adminDisputeService.markRefunded(principal.userId, disputeId)
    }

    @PostMapping("/{disputeId}/deny")
    fun deny(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable disputeId: UUID,
        @Valid @RequestBody body: DenyDisputeRequest,
    ): AdminDisputeDto = adminDisputeService.deny(principal.userId, disputeId, body.reason)

    @PostMapping("/{disputeId}/escalate")
    fun escalate(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable disputeId: UUID,
        @Valid @RequestBody body: EscalateDisputeRequest,
    ): AdminDisputeDto = adminDisputeService.escalate(principal.userId, disputeId, body.reason)
}
