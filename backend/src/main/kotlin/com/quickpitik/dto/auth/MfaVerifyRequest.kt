package com.quickpitik.dto.auth

import jakarta.validation.constraints.NotBlank

// Second step of an admin MFA login: the mfaPending token from /auth/login plus
// a 6-digit TOTP code (or a recovery code).
data class MfaVerifyRequest(
    @field:NotBlank
    val mfaToken: String,
    @field:NotBlank
    val code: String,
)
