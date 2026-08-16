package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ValidationException
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// `@Size(min = 8)` is a length floor, nothing more — "12345678" cleared it.
// The gate screens known-weak strings rather than imposing character classes,
// so anything the client forms accept still passes unless it's guessable.
class PasswordValidatorTest {

    @Test
    fun `an ordinary password passes`() {
        PasswordValidator.validate("marathon-cebu-2026", "password")
        PasswordValidator.validate("correct horse battery", "password")
    }

    @Test
    fun `the exact case the audit named is rejected`() {
        val ex = assertFailsWith<ValidationException> {
            PasswordValidator.validate("12345678", "password")
        }

        assertEquals(ErrorCodes.WEAK_PASSWORD, ex.code)
        assertEquals("password", ex.field)
    }

    @Test
    fun `common passwords are rejected regardless of case`() {
        assertFailsWith<ValidationException> { PasswordValidator.validate("password", "password") }
        assertFailsWith<ValidationException> { PasswordValidator.validate("PassWord", "password") }
        assertFailsWith<ValidationException> { PasswordValidator.validate("QWERTY123", "password") }
        assertFailsWith<ValidationException> { PasswordValidator.validate("changeme123", "password") }
    }

    @Test
    fun `a single repeated character is rejected`() {
        assertFailsWith<ValidationException> { PasswordValidator.validate("aaaaaaaa", "newPassword") }
        assertFailsWith<ValidationException> { PasswordValidator.validate("00000000", "newPassword") }
    }

    @Test
    fun `sequential runs are rejected in both directions`() {
        assertFailsWith<ValidationException> { PasswordValidator.validate("abcdefgh", "newPassword") }
        assertFailsWith<ValidationException> { PasswordValidator.validate("87654321", "newPassword") }
    }

    @Test
    fun `the reported field follows the call site`() {
        val ex = assertFailsWith<ValidationException> {
            PasswordValidator.validate("abcdefgh", "newPassword")
        }

        assertEquals("newPassword", ex.field)
    }

    // bcrypt hashes at most 72 bytes and discards the rest, so without a cap
    // two different long passwords hash the same and both authenticate.
    // Added 2026-08-16 as one leg of a three-module change — the website and
    // mobile forms reject over-long input first (validateNewPassword).

    @Test
    fun `a password at the bcrypt limit still passes`() {
        PasswordValidator.validate("a".repeat(71) + "z", "newPassword")
    }

    @Test
    fun `a password past the bcrypt limit is rejected`() {
        val ex = assertFailsWith<ValidationException> {
            PasswordValidator.validate("a".repeat(72) + "z", "newPassword")
        }

        assertEquals(ErrorCodes.WEAK_PASSWORD, ex.code)
        assertEquals("newPassword", ex.field)
    }

    @Test
    fun `the limit is counted in bytes, not characters`() {
        // Two alternating two-byte characters, so neither the repeated-character
        // nor the sequential-run rule fires and the length rule is what's under
        // test. 40 chars = 80 bytes: a naive character count would allow it,
        // but bcrypt would still truncate.
        assertFailsWith<ValidationException> {
            PasswordValidator.validate("éà".repeat(20), "newPassword")
        }
        // 36 chars = 72 bytes exactly, which fits.
        PasswordValidator.validate("éà".repeat(18), "newPassword")
    }
}
