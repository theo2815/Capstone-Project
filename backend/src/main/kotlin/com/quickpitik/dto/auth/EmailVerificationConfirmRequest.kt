package com.quickpitik.dto.auth

import jakarta.validation.constraints.NotBlank

/**
 * `POST /auth/verify-email`. Public, because the link is opened from a mail
 * client and the browser that follows it often carries no session — same shape
 * and same reasoning as [com.quickpitik.dto.profile.EmailChangeConfirmRequest].
 * The opaque token in the body is the whole credential.
 */
data class EmailVerificationConfirmRequest(
    @field:NotBlank(message = "token is required")
    val token: String,
)
