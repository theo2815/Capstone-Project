package com.quickpitik.dto.profile

import jakarta.validation.constraints.Email
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size

/**
 * `PUT /me/email` — step 1 of 2. Starts a change of the account's sign-in
 * email; nothing moves until the new address confirms.
 *
 * The current password is required because this is a credential change, not a
 * profile edit: whoever controls the sign-in address controls password reset,
 * so a hijacked session must not be able to walk the account away on its own.
 */
data class EmailChangeRequest(
    @field:NotBlank(message = "newEmail is required")
    @field:Email(message = "newEmail must be a valid email address")
    @field:Size(max = 255, message = "newEmail must be at most 255 characters")
    val newEmail: String,

    @field:NotBlank(message = "currentPassword is required")
    val currentPassword: String,
)

/**
 * `POST /auth/confirm-email-change` — step 2 of 2. Public, because the link is
 * opened from the NEW inbox, which may well be a browser with no session.
 */
data class EmailChangeConfirmRequest(
    @field:NotBlank(message = "token is required")
    val token: String,
)
