package com.quickpitik.controller

import com.quickpitik.dto.auth.AuthResponse
import com.quickpitik.dto.auth.ForgotPasswordRequest
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.auth.LogoutRequest
import com.quickpitik.dto.auth.RefreshRequest
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.dto.auth.ResetPasswordRequest
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.AuthService
import com.quickpitik.service.PasswordResetService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import jakarta.servlet.http.HttpServletRequest
import jakarta.validation.Valid
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/v1/auth")
class AuthController(
    private val authService: AuthService,
    private val passwordResetService: PasswordResetService,
    private val rateLimiter: RateLimiter,
) {
    @PostMapping("/register")
    fun register(
        @Valid @RequestBody req: RegisterRequest,
        request: HttpServletRequest,
    ): AuthResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_REGISTER, clientIp(request))
        return authService.register(req)
    }

    @PostMapping("/login")
    fun login(
        @Valid @RequestBody req: LoginRequest,
        request: HttpServletRequest,
    ): AuthResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, clientIp(request))
        return authService.login(req)
    }

    // Refresh is intentionally NOT rate-limited per IP — a single user
    // behind a shared NAT (campus, café) can legitimately fire refreshes
    // when access tokens expire. Abuse is bounded by refresh-token rotation
    // (parent token revoked on every refresh) + revoke-on-reuse.
    @PostMapping("/refresh")
    fun refresh(@Valid @RequestBody req: RefreshRequest): AuthResponse =
        authService.refresh(req)

    @PostMapping("/logout")
    fun logout(@RequestBody(required = false) req: LogoutRequest?): Map<String, Boolean> {
        authService.logout(req?.refreshToken)
        return mapOf("loggedOut" to true)
    }

    @PostMapping("/forgot-password")
    fun forgotPassword(
        @Valid @RequestBody req: ForgotPasswordRequest,
        request: HttpServletRequest,
    ): Map<String, String> {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_FORGOT_PASSWORD, clientIp(request))
        passwordResetService.requestReset(req.email)
        return mapOf("message" to "If that email exists, a reset link has been sent.")
    }

    @PostMapping("/reset-password")
    fun resetPassword(
        @Valid @RequestBody req: ResetPasswordRequest,
        request: HttpServletRequest,
    ): Map<String, String> {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_RESET_PASSWORD, clientIp(request))
        passwordResetService.confirmReset(req.token, req.newPassword)
        return mapOf("message" to "Password reset successful.")
    }

    @GetMapping("/me")
    fun me(@AuthenticationPrincipal principal: AuthPrincipal): UserDto =
        authService.me(principal)

    // X-Forwarded-For wins when present (behind a load balancer); fall back
    // to the direct remote address for local dev. Mirrors the helper in
    // PublicPhotographerController.
    private fun clientIp(request: HttpServletRequest): String {
        val forwarded = request.getHeader("X-Forwarded-For")
        if (!forwarded.isNullOrBlank()) {
            return forwarded.split(",").first().trim()
        }
        return request.remoteAddr ?: "unknown"
    }
}
