package com.quickpitik.controller

import com.quickpitik.dto.auth.ReauthRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.auth.StepUpService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

// Step-up re-auth: any authenticated user re-enters their password to obtain a
// short-lived step-up token for a sensitive action. Rate-limited on the login
// policy (keyed by user) so a hijacked session can't brute-force the password.
@RestController
@RequestMapping("/api/v1/me")
@PreAuthorize("isAuthenticated()")
class MeReauthController(
    private val stepUpService: StepUpService,
    private val rateLimiter: RateLimiter,
) {
    @PostMapping("/reauth")
    fun reauth(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: ReauthRequest,
    ): Map<String, String> {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, principal.userId.toString())
        return mapOf("stepUpToken" to stepUpService.reauth(principal.userId, body.password))
    }
}
