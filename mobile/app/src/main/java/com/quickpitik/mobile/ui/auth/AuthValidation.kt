package com.quickpitik.mobile.ui.auth

// Verbatim port of website/src/lib/auth-validation.ts. The Build Mandate
// requires mobile to apply the SAME validation rules as the website, so the
// regex, the limits, and the error copy are copied rather than re-derived —
// a runner who mistypes an email must read the same sentence on both surfaces.
//
// These are pre-submit hard gates only; the backend validates independently
// (@Email/@NotBlank on ForgotPasswordRequest, @Size(min = 8) on
// ResetPasswordRequest). Returns null when the value is acceptable.
//
// validateName() is not ported — mobile has no consumer for it.

private val EMAIL_RE = Regex("""^[^\s@]+@[^\s@]+\.[^\s@]+$""")

const val PASSWORD_MIN = 8
const val EMAIL_MAX = 254

/**
 * BCrypt hashes at most 72 bytes and silently discards the rest, so without a
 * cap two visually different long passwords hash identically and both succeed
 * at login. Measured in UTF-8 BYTES because that is the actual bcrypt limit —
 * an ASCII password hits it at 72 characters, a multi-byte one sooner. The
 * message says "characters" because that is what a user counts; the early trip
 * on multi-byte input is a safe failure, not a silent collision.
 *
 * Enforced by [validateNewPassword] (register / reset / change) and NEVER by
 * [validatePassword] (login), which must keep accepting whatever an existing
 * account was created with.
 * Backend counterpart: `PasswordValidator`. Website: `lib/auth-validation.ts`.
 */
const val PASSWORD_MAX_BYTES = 72

fun validateEmail(value: String): String? {
    val trimmed = value.trim()
    if (trimmed.isEmpty()) return "Email is required."
    if (trimmed.length > EMAIL_MAX) return "Email is limited to $EMAIL_MAX characters."
    if (!EMAIL_RE.matches(trimmed)) return "Use a valid email address."
    return null
}

/** The 6-digit password-reset OTP. Website counterpart: `lib/auth-validation.ts`. */
fun validateResetCode(value: String): String? {
    if (!Regex("""^\d{6}$""").matches(value)) return "Enter the 6-digit code from your email."
    return null
}

/** Sign-in gate. Length floor only — NEVER cap here, see [PASSWORD_MAX_BYTES]. */
fun validatePassword(value: String): String? {
    if (value.isEmpty()) return "Password is required."
    if (value.length < PASSWORD_MIN) return "Password must be at least $PASSWORD_MIN characters."
    return null
}

/**
 * Gate for a password being SET — register, reset, change. Adds the bcrypt
 * ceiling on top of [validatePassword]'s floor. Kept separate precisely so the
 * cap can never reach the login form, which must go on accepting whatever an
 * existing account was created with.
 */
fun validateNewPassword(value: String): String? {
    validatePassword(value)?.let { return it }
    if (value.toByteArray(Charsets.UTF_8).size > PASSWORD_MAX_BYTES)
        return "Password is limited to $PASSWORD_MAX_BYTES characters."
    return null
}
