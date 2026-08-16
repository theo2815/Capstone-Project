package com.quickpitik.dto.auth

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size

data class ResetPasswordRequest(
    @field:NotBlank
    val token: String,

    @field:NotBlank
    // 72 = bcrypt's truncation point; see PasswordValidator.MAX_BYTES.
    @field:Size(min = 8, max = 72, message = "Password must be 8-72 characters")
    val newPassword: String,
)
