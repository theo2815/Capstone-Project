package com.quickpitik.service.email

import com.quickpitik.config.ResendProperties
import org.junit.jupiter.api.Test
import org.springframework.http.HttpStatus
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

// Runner-audit "Checkout + webhook + email" #p1. A 429 used to propagate on the
// first try, leaving the receipt to PayMongo's webhook retry — which for an
// already-completed payment frequently never re-fires, so the buyer simply
// never got their download link.
//
// The send loop's HTTP path has no test harness in this module (PaymongoClient
// and AiApiClient share the same untested RestClient shape), so what is pinned
// here are the two decisions that actually govern the loop: which statuses are
// worth retrying, and how long to wait.
class ResendClientRetryTest {

    private val client = ResendClient(ResendProperties())

    @Test
    fun `rate limiting is retryable`() {
        // The whole point: Resend's free tier is ~2 req/s and a race finish
        // sends a burst of checkouts through in seconds.
        assertTrue(client.isRetryable(HttpStatus.TOO_MANY_REQUESTS))
    }

    @Test
    fun `upstream server errors are retryable`() {
        assertTrue(client.isRetryable(HttpStatus.INTERNAL_SERVER_ERROR))
        assertTrue(client.isRetryable(HttpStatus.BAD_GATEWAY))
        assertTrue(client.isRetryable(HttpStatus.SERVICE_UNAVAILABLE))
    }

    @Test
    fun `our own 4xx mistakes are not retryable`() {
        // A bad API key or an unverified sender fails identically every time —
        // retrying only delays the error and spends quota.
        assertFalse(client.isRetryable(HttpStatus.UNAUTHORIZED))
        assertFalse(client.isRetryable(HttpStatus.FORBIDDEN))
        assertFalse(client.isRetryable(HttpStatus.UNPROCESSABLE_ENTITY))
    }

    @Test
    fun `Retry-After in seconds is honoured`() {
        assertEquals(2_000L, client.retryAfterMs("2"))
    }

    @Test
    fun `an absurd Retry-After is capped rather than pinning a pool thread`() {
        assertEquals(5_000L, client.retryAfterMs("600"))
    }

    @Test
    fun `a missing or unparseable Retry-After falls back to the backoff ladder`() {
        // Null means "no opinion" — the caller then uses its own backoff.
        assertNull(client.retryAfterMs(null))
        assertNull(client.retryAfterMs("Wed, 21 Oct 2026 07:28:00 GMT"))
        assertNull(client.retryAfterMs(""))
    }

    @Test
    fun `a zero or negative Retry-After is ignored`() {
        assertNull(client.retryAfterMs("0"))
        assertNull(client.retryAfterMs("-5"))
    }

    @Test
    fun `surrounding whitespace does not defeat the parse`() {
        assertEquals(3_000L, client.retryAfterMs("  3  "))
    }
}
