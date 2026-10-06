package com.quickpitik.controller

import com.quickpitik.dto.removal.*
import com.quickpitik.entity.RemovalRequestStatus
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import com.quickpitik.service.removal.PhotoRemovalQueryService
import com.quickpitik.service.removal.PhotoRemovalService
import jakarta.servlet.http.HttpServletResponse
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.*
import java.util.UUID

@RestController
@RequestMapping("/api/v1/me/photographer/photo-removal-requests")
@PreAuthorize("hasRole('PHOTOGRAPHER')")
class MePhotographerPhotoRemovalController(
    private val service: PhotoRemovalService,
    private val queries: PhotoRemovalQueryService,
    private val rateLimiter: RateLimiter,
) {
    @GetMapping
    fun list(@AuthenticationPrincipal principal: AuthPrincipal, @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?, @RequestParam(required = false) eventId: UUID?,
        @RequestParam(required = false) status: RemovalRequestStatus?, @RequestParam(required = false) needsAction: Boolean?,
        response: HttpServletResponse): RemovalPage<PhotographerRemovalRequestDto> {
        limit(principal, response)
        return queries.photographerList(principal.userId, offset, limit, eventId, status, needsAction)
    }

    @GetMapping("/{id}")
    fun detail(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable id: UUID,
        response: HttpServletResponse): PhotographerRemovalRequestDto {
        limit(principal, response)
        return queries.photographerDetail(principal.userId, id)
    }

    @PostMapping("/{id}/responses")
    fun respond(@AuthenticationPrincipal principal: AuthPrincipal, @PathVariable id: UUID,
        @Valid @RequestBody command: RespondToRemovalCommand, response: HttpServletResponse): PhotographerRemovalRequestDto {
        limit(principal, response, Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD)
        return service.respond(principal.userId, id, command)
    }

    private fun limit(principal: AuthPrincipal, response: HttpServletResponse, policy: String = Bucket4jRateLimiter.POLICY_ORDER_READ) {
        response.setHeader("Cache-Control", "private, no-store")
        rateLimiter.acquireOrThrow(policy, principal.userId.toString())
    }
}
