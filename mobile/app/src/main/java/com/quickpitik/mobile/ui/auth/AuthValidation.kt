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

fun validateEmail(value: String): String? {
    val trimmed = value.trim()
    if (trimmed.isEmpty()) return "Email is required."
    if (trimmed.length > EMAIL_MAX) return "Email is limited to $EMAIL_MAX characters."
    if (!EMAIL_RE.matches(trimmed)) return "Use a valid email address."
    return null
}

fun validatePassword(value: String): String? {
    if (value.isEmpty()) return "Password is required."
    if (value.length < PASSWORD_MIN) return "Password must be at least $PASSWORD_MIN characters."
    return null
}
