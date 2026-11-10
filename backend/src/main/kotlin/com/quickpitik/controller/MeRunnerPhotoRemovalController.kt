package com.quickpitik.controller

import com.quickpitik.dto.photographer.LivenessSessionDto
import com.quickpitik.dto.removal.RunnerRemovalEligibilityDto
import com.quickpitik.dto.removal.RunnerRemovalReason
import com.quickpitik.dto.removal.RunnerRemovalReasonDto
import com.quickpitik.dto.removal.StartRunnerLivenessCommand
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import com.quickpitik.service.removal.RunnerPhotoRemovalService
import com.quickpitik.service.removal.RunnerRemovalEligibilityService
import jakarta.servlet.http.HttpServletResponse
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

// A runner's photo deletion requests (RD-2: reasons, eligibility, start the live check).
@RestController
@RequestMapping("/api/v1/me/runner/photo-removal-requests")
@PreAuthorize("hasRole('RUNNER')")
class MeRunnerPhotoRemovalController(
    private val eligibilityService: RunnerRemovalEligibilityService,
    private val service: RunnerPhotoRemovalService,
    private val rateLimiter: RateLimiter,
) {
    @GetMapping("/reasons")
    fun reasons(@AuthenticationPrincipal principal: AuthPrincipal, response: HttpServletResponse): List<RunnerRemovalReasonDto> {
        read(principal, response)
        return RunnerRemovalReason.entries.map { RunnerRemovalReasonDto(it.name, it.label) }
    }

    @GetMapping("/eligibility")
    fun eligibility(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestParam eventId: UUID,
        response: HttpServletResponse,
    ): RunnerRemovalEligibilityDto {
        read(principal, response)
        val r = eligibilityService.check(principal.userId, eventId)
        return RunnerRemovalEligibilityDto(r.eligible, r.blockedReason?.name, r.attemptsLeftToday, r.emailVerified, r.openRequestId)
    }

    // The response carries live AWS keys: never cached, never logged.
    @PostMapping("/liveness-session")
    fun startLiveness(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody command: StartRunnerLivenessCommand,
        response: HttpServletResponse,
    ): LivenessSessionDto {
        response.setHeader("Cache-Control", "private, no-store")
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_PHOTO_SEARCH, "removal:${principal.userId}")
        return service.startLiveness(principal.userId, command)
    }

    private fun read(principal: AuthPrincipal, response: HttpServletResponse) {
        response.setHeader("Cache-Control", "private, no-store")
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_ORDER_READ, "removal-read:${principal.userId}")
    }
}
