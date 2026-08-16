package com.quickpitik.dto.profile

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size

data class PasswordChangeRequest(
    @field:NotBlank(message = "currentPassword is required")
    val currentPassword: String,

    @field:NotBlank(message = "newPassword is required")
    // 72 = bcrypt's truncation point; see PasswordValidator.MAX_BYTES.
    @field:Size(min = 8, max = 72, message = "newPassword must be 8-72 characters")
    val newPassword: String,

    /**
     * The caller's CURRENT refresh token. Every other session is revoked on a
     * successful change; the token sent here is the one spared.
     *
     * Leaving this null or blank revokes **everything, including the caller's
     * own session** — the client is signed out of the device it just changed
     * the password on. That fallback is deliberate (it keeps the endpoint
     * usable for clients that can't supply a refresh token), but any normal
     * client should send its stored token. See `ProfileService`'s KDoc.
     */
    val refreshToken: String? = null,
)
