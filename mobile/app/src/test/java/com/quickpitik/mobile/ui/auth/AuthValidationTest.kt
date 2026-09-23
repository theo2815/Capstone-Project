package com.quickpitik.mobile.ui.auth

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The pre-submit gates ported verbatim from the website's auth-validation.ts.
 * Messages are asserted because the Build Mandate promises the same sentence on
 * both surfaces. The bcrypt ceiling is asserted in BYTES: that is the hash's
 * real limit, and a multi-byte password is meant to trip it early rather than
 * collide silently with a different password after truncation.
 */
class AuthValidationTest {
    @Test fun `email needs a value, a shape and at most 254 characters`() {
        assertEquals("Email is required.", validateEmail("   "))
        assertEquals("Use a valid email address.", validateEmail("juan@"))
        assertEquals("Use a valid email address.", validateEmail("juan@cebu"))
        assertEquals("Use a valid email address.", validateEmail("ju an@x.com"))
        assertNull(validateEmail("  juan@x.com  "))
        assertNull(validateEmail("a".repeat(249) + "@x.co"))
        assertEquals("Email is limited to 254 characters.", validateEmail("a".repeat(250) + "@x.co"))
    }

    @Test fun `reset code is exactly six digits`() {
        assertNull(validateResetCode("123456"))
        for (bad in listOf("", "12345", "1234567", "12345a", " 123456")) {
            assertEquals(bad, "Enter the 6-digit code from your email.", validateResetCode(bad))
        }
    }

    @Test fun `login password has a floor and deliberately no ceiling`() {
        assertEquals("Password is required.", validatePassword(""))
        assertEquals("Password must be at least 8 characters.", validatePassword("1234567"))
        assertNull(validatePassword("12345678"))
        // An account created before the cap must keep logging in.
        assertNull(validatePassword("x".repeat(200)))
    }

    @Test fun `new password caps at 72 bytes, not 72 characters`() {
        assertEquals("Password must be at least 8 characters.", validateNewPassword("short"))
        assertNull(validateNewPassword("a".repeat(72)))
        assertEquals("Password is limited to 72 characters.", validateNewPassword("a".repeat(73)))
        // 24 euro signs are 72 UTF-8 bytes; 25 are 75, refused although only 25 characters.
        assertNull(validateNewPassword("€".repeat(24)))
        assertEquals("Password is limited to 72 characters.", validateNewPassword("€".repeat(25)))
    }
}
