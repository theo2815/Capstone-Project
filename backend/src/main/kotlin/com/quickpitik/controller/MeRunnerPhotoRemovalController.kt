package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.IdempotencyKey
import com.quickpitik.dto.photographer.LivenessSessionDto
import com.quickpitik.dto.removal.RunnerRemovalEligibilityDto
import com.quickpitik.dto.removal.RunnerRemovalReason
import com.quickpitik.dto.removal.RunnerRemovalReasonDto
import com.quickpitik.dto.removal.RunnerSubmitResultDto
import com.quickpitik.dto.removal.StartRunnerLivenessCommand
import com.quickpitik.dto.removal.SubmitRunnerRemovalCommand
import com.quickpitik.exception.ValidationException
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import com.quickpitik.service.removal.RunnerPhotoRemovalService
import com.quickpitik.service.removal.RunnerRemovalEligibilityService
import jakarta.servlet.http.HttpServletResponse
import jakarta.validation.Valid
import org.springframework.http.HttpStatus
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestHeader
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.ResponseStatus
import org.springframework.web.bind.annotation.RestController
import java.util.UUID

// A runner's photo deletion requests (RD-2: reasons, eligibility, start the live check; RD-4: submit).
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

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    fun submit(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestHeader(IdempotencyKey.HEADER, required = false) idempotencyKey: String?,
        @Valid @RequestBody command: SubmitRunnerRemovalCommand,
        response: HttpServletResponse,
    ): RunnerSubmitResultDto {
        response.setHeader("Cache-Control", "private, no-store")
        val key = IdempotencyKey.parse(idempotencyKey) ?: throw ValidationException(
            code = ErrorCodes.INVALID_IDEMPOTENCY_KEY, message = "Idempotency-Key is required", field = IdempotencyKey.HEADER)
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_PHOTO_SEARCH, "removal:${principal.userId}")
        return service.submit(principal.userId, UUID.fromString(key.value), command)
    }

    private fun read(principal: AuthPrincipal, response: HttpServletResponse) {
        response.setHeader("Cache-Control", "private, no-store")
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_ORDER_READ, "removal-read:${principal.userId}")
    }
}
