package com.quickpitik.service.admin

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// Correctness pinned to the RFC 6238 Appendix B test vectors (SHA-1, 8 digits,
// ASCII secret "12345678901234567890"). If the HMAC/truncation is wrong these fail.
class TotpServiceTest {

    private val service = TotpService()
    private val secret = "12345678901234567890".toByteArray()

    @Test
    fun `matches RFC 6238 SHA-1 vectors`() {
        assertEquals("94287082", service.generate(secret, 59L, digits = 8))
        assertEquals("07081804", service.generate(secret, 1111111109L, digits = 8))
        assertEquals("14050471", service.generate(secret, 1111111111L, digits = 8))
        assertEquals("89005924", service.generate(secret, 1234567890L, digits = 8))
        assertEquals("69279037", service.generate(secret, 2000000000L, digits = 8))
        assertEquals("65353130", service.generate(secret, 20000000000L, digits = 8))
    }

    @Test
    fun `verify accepts the current code and one step of skew`() {
        val now = 1_700_000_000L
        assertTrue(service.verify(secret, service.generate(secret, now), now))
        // Code from the previous 30s window is still accepted (skew).
        assertTrue(service.verify(secret, service.generate(secret, now - 30), now))
        assertTrue(service.verify(secret, service.generate(secret, now + 30), now))
    }

    @Test
    fun `verify rejects a wrong or two-steps-stale code`() {
        val now = 1_700_000_000L
        assertFalse(service.verify(secret, "000000", now))
        assertFalse(service.verify(secret, service.generate(secret, now - 90), now))
        assertFalse(service.verify(secret, "12345", now)) // wrong length
    }

    @Test
    fun `base32 encodes per RFC 4648 (unpadded, uppercase)`() {
        // "foobar" -> MZXW6YTBOI (RFC 4648 test vector, padding stripped)
        assertEquals("MZXW6YTBOI", service.base32("foobar".toByteArray()))
    }
}
