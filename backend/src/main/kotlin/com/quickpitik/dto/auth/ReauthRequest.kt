package com.quickpitik.dto.auth

import jakarta.validation.constraints.NotBlank

// POST /me/reauth — the user re-enters their current password to obtain a
// short-lived step-up token for a sensitive action.
data class ReauthRequest(
    @field:NotBlank(message = "password is required")
    val password: String,
)
