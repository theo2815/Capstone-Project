package com.quickpitik.mobile.data.remote

data class SelfieRefDto(
    val id: String,
    val dataUrl: String,
    val uploadedAt: String,
    val isPrimary: Boolean,
    // 0..1, and 0 for EVERY selfie while AI_API_ENABLED=false — the quality
    // gate is skipped, not failed. Do not render it as a score on its own;
    // read [qualityTestStatus] to know whether the number means anything.
    val qualityScore: Double,
    // "untested" | "passed" (backend V26). "untested" = uploaded while ai-api
    // was off, so it has never been checked and may not match once search goes
    // live. "rejected" is unreachable: the gate throws before the row is saved,
    // so a rejected selfie is never persisted. Defaulted for the same reason
    // cleanUrl is — an older backend simply omits the field.
    val qualityTestStatus: String = "untested"
)

data class ProfileUpdateRequest(
    val name: String
)

/**
 * Body for `PUT /me/email` — step 1 of 2. Mirrors backend
 * dto/profile/EmailChangeRequest.
 *
 * This does NOT change the sign-in email. It mails a confirmation link to
 * [newEmail], and nothing moves until that link is redeemed from that inbox —
 * so the UI must never report success as "email updated".
 *
 * Redemption is web-only: `EmailService` builds the link against the website
 * origin, so there is no mobile screen for step 2. Same shape as the pasted
 * password-reset token — the deliberate deviation is documented in the
 * 2026-08-14 auth-recovery ADR.
 *
 * [currentPassword] is required because this is a credential change: whoever
 * controls the sign-in address controls password reset.
 */
data class EmailChangeRequest(
    val newEmail: String,
    val currentPassword: String
)

data class PasswordChangeRequest(
    val currentPassword: String,
    val newPassword: String,
    val refreshToken: String? = null
)
