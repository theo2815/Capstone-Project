package com.quickpitik.dto.auth

import com.quickpitik.entity.Role
import jakarta.validation.constraints.NotBlank

// `role` is null on the first attempt; a brand-new Google user gets 422
// ROLE_REQUIRED back and the client re-POSTs the same idToken with the
// picked role. Existing users ignore the field entirely.
data class GoogleLoginRequest(
    @field:NotBlank
    val idToken: String,
    val role: Role? = null,
)
