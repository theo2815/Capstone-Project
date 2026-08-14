package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ValidationException

/**
 * Shared weak-password gate for register / change-password / reset-password.
 * `@Size(min = 8)` on the DTOs is a length floor only — it happily accepts
 * "12345678".
 *
 * Deliberately NOT a character-class rule. Both clients validate length only
 * (website `lib/auth-validation.ts`, mobile `ui/auth/AuthValidation.kt`), so a
 * composition requirement here would reject passwords their forms told the user
 * were fine, and would need a coordinated three-way release to fix. This follows
 * NIST SP 800-63B instead: no composition rules, screen against known-weak
 * strings. Everything the client forms accept today still passes unless it is
 * genuinely guessable.
 *
 * Stateless object rather than an injected bean — same shape as
 * `service/image/ExifOrientation.kt`. There is nothing to inject.
 */
object PasswordValidator {

    fun validate(password: String, field: String) {
        val normalized = password.lowercase()
        if (normalized in COMMON_PASSWORDS) {
            reject(field, "That password is too common — pick something harder to guess.")
        }
        if (isSingleRepeatedCharacter(normalized)) {
            reject(field, "Password can't be the same character repeated.")
        }
        if (isSequentialRun(normalized)) {
            reject(field, "Password can't be a simple sequence like 12345678 or abcdefgh.")
        }
    }

    private fun reject(field: String, message: String): Nothing =
        throw ValidationException(message, ErrorCodes.WEAK_PASSWORD, field)

    private fun isSingleRepeatedCharacter(value: String): Boolean =
        value.isNotEmpty() && value.all { it == value[0] }

    /** True for a run of consecutive code points in either direction: "12345678", "hgfedcba". */
    private fun isSequentialRun(value: String): Boolean {
        if (value.length < 2) return false
        val step = value[1].code - value[0].code
        if (step != 1 && step != -1) return false
        return value.zipWithNext().all { (a, b) -> b.code - a.code == step }
    }

    // Screening list, not an exhaustive breach corpus — the top repeat offenders
    // from public breach dumps that clear the 8-character floor, plus this
    // project's own guessables (the brand name and the dev bootstrap default).
    private val COMMON_PASSWORDS = setOf(
        "password",
        "password1",
        "password12",
        "password123",
        "passw0rd",
        "p@ssw0rd",
        "12345678",
        "123456789",
        "1234567890",
        "qwerty123",
        "qwertyui",
        "qwertyuiop",
        "1q2w3e4r",
        "iloveyou",
        "princess",
        "football",
        "baseball",
        "sunshine",
        "superman",
        "trustno1",
        "welcome1",
        "welcome123",
        "letmein1",
        "letmein123",
        "abc12345",
        "abcd1234",
        "admin123",
        "administrator",
        "changeme",
        "changeme123",
        "quickpitik",
        "quickpitik1",
        "quickpitik123",
    )
}
