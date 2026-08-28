package com.quickpitik.controller

import com.quickpitik.dto.auth.AuthResponse
import com.quickpitik.dto.auth.EmailVerificationConfirmRequest
import com.quickpitik.dto.auth.ForgotPasswordRequest
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.auth.LogoutRequest
import com.quickpitik.dto.auth.MessageResponse
import com.quickpitik.dto.auth.RefreshRequest
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.dto.auth.ResetPasswordRequest
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.dto.auth.VerifyResetOtpRequest
import com.quickpitik.dto.auth.VerifyResetOtpResponse
import com.quickpitik.dto.profile.EmailChangeConfirmRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.AuthService
import com.quickpitik.service.EmailVerificationService
import com.quickpitik.service.PasswordResetService
import com.quickpitik.service.profile.EmailChangeService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import com.quickpitik.service.ratelimit.clientIp
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
    private val emailChangeService: EmailChangeService,
    private val emailVerificationService: EmailVerificationService,
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
    ): MessageResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_FORGOT_PASSWORD, clientIp(request))
        passwordResetService.requestReset(req.email)
        return MessageResponse("If that email exists, a reset code has been sent.")
    }

    // Step 2 of the OTP reset flow: trades the mailed 6-digit code for the
    // one-shot continuation token /auth/reset-password consumes. Public — the
    // caller is by definition signed out. Its own policy (not reset-password's)
    // so code guessing can't drain the confirm step's budget, whose token is
    // unguessable anyway.
    @PostMapping("/verify-reset-otp")
    fun verifyResetOtp(
        @Valid @RequestBody req: VerifyResetOtpRequest,
        request: HttpServletRequest,
    ): VerifyResetOtpResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_VERIFY_RESET_OTP, clientIp(request))
        return VerifyResetOtpResponse(passwordResetService.verifyOtp(req.email, req.code))
    }

    @PostMapping("/reset-password")
    fun resetPassword(
        @Valid @RequestBody req: ResetPasswordRequest,
        request: HttpServletRequest,
    ): MessageResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_RESET_PASSWORD, clientIp(request))
        passwordResetService.confirmReset(req.token, req.newPassword)
        return MessageResponse("Password reset successful.")
    }

    // Step 2 of the change-email flow. Public because the link is opened from
    // the NEW inbox, which is very often a browser with no QuickPitik session —
    // same reason /reset-password is public. The opaque token is the credential.
    //
    // Shares the reset-password rate policy on purpose: identical threat shape
    // (unauthenticated, token in the body, so a guessing surface), and no reason
    // to give an attacker a second independent budget.
    @PostMapping("/confirm-email-change")
    fun confirmEmailChange(
        @Valid @RequestBody req: EmailChangeConfirmRequest,
        request: HttpServletRequest,
    ): MessageResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_RESET_PASSWORD, clientIp(request))
        emailChangeService.confirmChange(req.token)
        return MessageResponse("Email updated. Sign in again with your new address.")
    }

    // Redeems the link mailed at registration. Public for the same reason
    // /confirm-email-change is — it's opened from an inbox, and the opaque
    // token is the credential. Shares the reset-password bucket: identical
    // threat shape (unauthenticated, guessable-token surface).
    //
    // Advisory: this stamps users.email_verified_at and nothing else. No
    // endpoint gates on it. See EmailVerificationService.
    @PostMapping("/verify-email")
    fun verifyEmail(
        @Valid @RequestBody req: EmailVerificationConfirmRequest,
        request: HttpServletRequest,
    ): MessageResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_RESET_PASSWORD, clientIp(request))
        emailVerificationService.confirm(req.token)
        return MessageResponse("Email confirmed. Thanks!")
    }

    // Deliberately NOT in SecurityConfig's permitAll list — register signs the
    // user straight in, so the one caller that needs this already has a bearer
    // token. Keyed by userId rather than IP for the same reason, and on the
    // forgot-password policy because the abuse shape is identical: mail sent to
    // a third party on demand.
    @PostMapping("/resend-verification")
    fun resendVerification(@AuthenticationPrincipal principal: AuthPrincipal): MessageResponse {
        rateLimiter.acquireOrThrow(
            Bucket4jRateLimiter.POLICY_AUTH_FORGOT_PASSWORD,
            principal.userId.toString(),
        )
        emailVerificationService.resend(principal.userId)
        return MessageResponse("Verification email sent. Check your inbox.")
    }

    @GetMapping("/me")
    fun me(@AuthenticationPrincipal principal: AuthPrincipal): UserDto =
        authService.me(principal)
}
