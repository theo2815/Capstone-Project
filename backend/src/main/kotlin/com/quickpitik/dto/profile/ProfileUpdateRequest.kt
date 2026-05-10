package com.quickpitik.dto.profile

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size

data class ProfileUpdateRequest(
    @field:NotBlank(message = "name is required")
    @field:Size(min = 1, max = 100, message = "name must be 1-100 characters")
    val name: String,
)
