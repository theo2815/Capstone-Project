package com.quickpitik.dto.admin

import jakarta.validation.constraints.NotBlank

// otpauth:// URI for the authenticator QR + the base32 secret for manual entry.
data class MfaEnrollResponse(val otpauthUri: String, val secret: String)

// Returned ONCE on confirm — the plaintext recovery codes are never stored.
data class MfaConfirmResponse(val recoveryCodes: List<String>)

data class MfaStatusResponse(val enabled: Boolean)

// A 6-digit TOTP code OR a recovery code (disable / login-verify accept either).
data class MfaCodeRequest(
    @field:NotBlank
    val code: String,
)
