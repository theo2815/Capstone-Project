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
) {
    @PostMapping("/register")
    fun register(@Valid @RequestBody req: RegisterRequest): AuthResponse =
        authService.register(req)

    @PostMapping("/login")
    fun login(@Valid @RequestBody req: LoginRequest): AuthResponse =
        authService.login(req)

    @PostMapping("/refresh")
    fun refresh(@Valid @RequestBody req: RefreshRequest): AuthResponse =
        authService.refresh(req)

    @PostMapping("/logout")
    fun logout(@RequestBody(required = false) req: LogoutRequest?): Map<String, Boolean> {
        authService.logout(req?.refreshToken)
        return mapOf("loggedOut" to true)
    }

    @PostMapping("/forgot-password")
    fun forgotPassword(@Valid @RequestBody req: ForgotPasswordRequest): Map<String, String> {
        passwordResetService.requestReset(req.email)
        return mapOf("message" to "If that email exists, a reset link has been sent.")
    }

    @PostMapping("/reset-password")
    fun resetPassword(@Valid @RequestBody req: ResetPasswordRequest): Map<String, String> {
        passwordResetService.confirmReset(req.token, req.newPassword)
        return mapOf("message" to "Password reset successful.")
    }

    @GetMapping("/me")
    fun me(@AuthenticationPrincipal principal: AuthPrincipal): UserDto =
        authService.me(principal)
}
