package com.quickpitik.controller

import com.quickpitik.dto.admin.MfaCodeRequest
import com.quickpitik.dto.admin.MfaConfirmResponse
import com.quickpitik.dto.admin.MfaEnrollResponse
import com.quickpitik.dto.admin.MfaStatusResponse
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminMfaService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

// Opt-in admin TOTP MFA management. Login enforcement is separate (Phase 5b).
@RestController
@RequestMapping("/api/v1/admin/mfa")
@PreAuthorize("hasRole('ADMIN')")
class AdminMfaController(
    private val adminMfaService: AdminMfaService,
    private val rateLimiter: RateLimiter,
) {
    @GetMapping("/status")
    fun status(@AuthenticationPrincipal principal: AuthPrincipal): MfaStatusResponse =
        adminMfaService.status(principal.userId)

    @PostMapping("/enroll")
    fun enroll(@AuthenticationPrincipal principal: AuthPrincipal): MfaEnrollResponse =
        adminMfaService.enroll(principal.userId)

    @PostMapping("/confirm")
    fun confirm(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: MfaCodeRequest,
    ): MfaConfirmResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, "admin-mfa:${principal.userId}")
        return adminMfaService.confirm(principal.userId, body.code)
    }

    @PostMapping("/disable")
    fun disable(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: MfaCodeRequest,
    ): Map<String, Boolean> {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, "admin-mfa:${principal.userId}")
        return adminMfaService.disable(principal.userId, body.code)
    }
}
