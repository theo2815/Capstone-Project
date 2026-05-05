package com.quickpitik.dto.auth

import jakarta.validation.constraints.Email
import jakarta.validation.constraints.NotBlank

data class ForgotPasswordRequest(
    @field:Email
    @field:NotBlank
    val email: String,
)
