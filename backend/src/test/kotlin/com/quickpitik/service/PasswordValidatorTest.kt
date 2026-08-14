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
}
