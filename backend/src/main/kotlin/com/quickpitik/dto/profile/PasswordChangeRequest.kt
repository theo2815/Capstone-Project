package com.quickpitik.dto.profile

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size

data class PasswordChangeRequest(
    @field:NotBlank(message = "currentPassword is required")
    val currentPassword: String,

    @field:NotBlank(message = "newPassword is required")
    @field:Size(min = 8, max = 128, message = "newPassword must be 8-128 characters")
    val newPassword: String,
)
