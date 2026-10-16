package com.quickpitik.controller

import com.quickpitik.dto.auth.AuthResponse
import com.quickpitik.dto.auth.DesktopCodeRequest
import com.quickpitik.dto.auth.DesktopCodeResponse
import com.quickpitik.dto.auth.DesktopExchangeRequest
import com.quickpitik.dto.auth.EmailVerificationConfirmRequest
import com.quickpitik.dto.auth.ForgotPasswordRequest
import com.quickpitik.dto.auth.GoogleLoginRequest
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.auth.LogoutRequest
import com.quickpitik.dto.auth.MessageResponse
import com.quickpitik.dto.auth.MfaVerifyRequest
import com.quickpitik.dto.auth.RefreshRequest
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.dto.auth.ResetPasswordRequest
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.dto.auth.VerifyResetOtpRequest
import com.quickpitik.dto.auth.VerifyResetOtpResponse
import com.quickpitik.dto.profile.EmailChangeConfirmRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.auth.AuthService
import com.quickpitik.service.auth.DesktopAuthService
import com.quickpitik.service.auth.EmailVerificationService
import com.quickpitik.service.auth.GoogleAuthService
import com.quickpitik.service.auth.PasswordResetService
import com.quickpitik.service.profile.EmailChangeService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import com.quickpitik.service.ratelimit.clientIp
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import jakarta.validation.Valid
import org.springframework.security.access.prepost.PreAuthorize
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
    private val googleAuthService: GoogleAuthService,
    private val passwordResetService: PasswordResetService,
    private val emailChangeService: EmailChangeService,
    private val emailVerificationService: EmailVerificationService,
    private val desktopAuthService: DesktopAuthService,
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

    // Second step of an admin MFA login. Shares the login rate policy — same
    // threat shape (a credential, here the code, in the body). Public: the
    // caller holds only the short-lived mfaPending token, not a session.
    @PostMapping("/mfa/verify")
    fun mfaVerify(
        @Valid @RequestBody req: MfaVerifyRequest,
        request: HttpServletRequest,
    ): AuthResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, clientIp(request))
        return authService.verifyMfa(req.mfaToken, req.code)
    }

    // "Continue with Google" — website (GIS button) and mobile (Credential
    // Manager) both exchange a Google-signed ID token here for the normal
    // QuickPitik pair. Shares the login policy: identical threat shape
    // (unauthenticated credential in the body), no reason for a second budget.
    @PostMapping("/google")
    fun google(
        @Valid @RequestBody req: GoogleLoginRequest,
        request: HttpServletRequest,
    ): AuthResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, clientIp(request))
        return googleAuthService.login(req)
    }

    // Refresh is intentionally NOT rate-limited per IP — a single user
    // behind a shared NAT (campus, café) can legitimately fire refreshes
    // when access tokens expire. Abuse is bounded by refresh-token rotation:
    // the parent is revoked on every refresh, and a parent re-presented within
    // jwt.refresh-rotation-grace (a lost answer, or a second browser tab) gets
    // one more token while its first child is still live. There is no
    // revoke-on-reuse of the whole family.
    @PostMapping("/refresh")
    fun refresh(@Valid @RequestBody req: RefreshRequest): AuthResponse =
        authService.refresh(req)

    // Desktop connect (V85), step 1: the signed-in website asks for a one-time
    // code bound to the BatchMyPhotos PKCE challenge. Photographers only; the
    // code is returned once and never cached.
    @PostMapping("/desktop/codes")
    @PreAuthorize("hasRole('PHOTOGRAPHER')")
    fun mintDesktopCode(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody req: DesktopCodeRequest,
        response: HttpServletResponse,
    ): DesktopCodeResponse {
        response.setHeader("Cache-Control", "no-store")
        return desktopAuthService.mint(principal.userId, req.codeChallenge)
    }

    // Step 2: the desktop redeems the code with its verifier. Public — the code
    // in the body is the credential — so it shares the login rate policy.
    @PostMapping("/desktop/exchange")
    fun exchangeDesktopCode(
        @Valid @RequestBody req: DesktopExchangeRequest,
        request: HttpServletRequest,
        response: HttpServletResponse,
    ): AuthResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, clientIp(request))
        response.setHeader("Cache-Control", "no-store")
        return desktopAuthService.exchange(req.code, req.codeVerifier)
    }

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
