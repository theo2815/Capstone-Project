package com.quickpitik.dto.auth

import jakarta.validation.constraints.NotBlank

data class RefreshRequest(
    @field:NotBlank
    val refreshToken: String,
)
