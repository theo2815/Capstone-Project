package com.quickpitik.service.ratelimit

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ApiException
import org.junit.jupiter.api.Test
import org.springframework.http.HttpStatus
import org.springframework.mock.web.MockHttpServletRequest
import java.time.Duration
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// acquireOrThrow is the one seam between every rate-limited controller and the
// bucket, and clientIp is the deliberate remoteAddr-not-X-Forwarded-For decision
// (2026-08-27). Neither had a test (2026-09-11 test-suite audit).
class RateLimitGateTest {

    private fun limiter(decision: RateLimiter.Decision) = object : RateLimiter {
        override fun tryAcquire(policy: String, key: String) = decision
    }

    @Test
    fun `a denied bucket answers 429 with a Retry-After the client can wait out`() {
        val ex = assertFailsWith<ApiException> {
            limiter(RateLimiter.Decision(allowed = false, retryAfter = Duration.ofMillis(1500)))
                .acquireOrThrow("auth-login", "203.0.113.9")
        }

        assertEquals(HttpStatus.TOO_MANY_REQUESTS, ex.status)
        assertEquals(ErrorCodes.RATE_LIMITED, ex.code)
        // Rounded up, never a wait the bucket has not actually finished.
        assertEquals(2L, ex.retryAfterSeconds)
    }

    @Test
    fun `a denied bucket with no refill wait still asks for at least one second`() {
        val ex = assertFailsWith<ApiException> {
            limiter(RateLimiter.Decision(allowed = false, retryAfter = Duration.ZERO))
                .acquireOrThrow("auth-login", "203.0.113.9")
        }

        assertEquals(1L, ex.retryAfterSeconds)
    }

    @Test
    fun `clientIp ignores a forged X-Forwarded-For`() {
        val request = MockHttpServletRequest().apply {
            remoteAddr = "203.0.113.9"
            addHeader("X-Forwarded-For", "198.51.100.1, 203.0.113.9")
        }

        assertEquals("203.0.113.9", clientIp(request))
    }
}
