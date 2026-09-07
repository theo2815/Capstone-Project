package com.quickpitik.service.admin

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MfaSecretCipherTest {

    private val cipher = MfaSecretCipher("a-sufficiently-long-dev-jwt-secret-for-tests-0123456789")

    @Test
    fun `round-trips a TOTP secret`() {
        val secret = TotpService().randomSecret()
        val restored = cipher.decrypt(cipher.encrypt(secret))
        assertTrue(secret.contentEquals(restored))
    }

    @Test
    fun `two encryptions of the same secret differ (fresh IV)`() {
        val secret = "12345678901234567890".toByteArray()
        assertFalse(cipher.encrypt(secret) == cipher.encrypt(secret))
    }

    @Test
    fun `a different key cannot decrypt`() {
        val other = MfaSecretCipher("a-completely-different-jwt-secret-value-9876543210zzzzzz")
        val token = cipher.encrypt("hello-totp".toByteArray())
        var failed = false
        try {
            other.decrypt(token)
        } catch (ex: Exception) {
            failed = true // GCM tag check must reject a wrong key
        }
        assertEquals(true, failed)
    }
}
