package com.quickpitik.dto.auth

// The short-lived one-shot continuation token minted by a successful OTP
// verification; it is the only credential /auth/reset-password accepts.
data class VerifyResetOtpResponse(
    val resetToken: String,
)
