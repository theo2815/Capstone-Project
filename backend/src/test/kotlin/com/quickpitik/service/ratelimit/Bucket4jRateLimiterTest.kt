package com.quickpitik.service.ratelimit

import com.quickpitik.config.RateLimitProperties
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import java.time.Duration
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// The token-bucket limiter itself (previously covered only indirectly via
// controller gate tests that mock the interface). Pins the NFR-S-11 window
// behavior, key isolation, the fail-loud unknown-policy contract, and the
// idle-bucket eviction sweep.
class Bucket4jRateLimiterTest {

    private fun policy(capacity: Long, period: Duration) =
        RateLimitProperties.Policy(capacity = capacity, refillPeriod = period)

    @Test
    fun `allows up to capacity then denies with a positive retry-after`() {
        val limiter = Bucket4jRateLimiter(
            RateLimitProperties(authLogin = policy(3, Duration.ofMinutes(15))),
            SimpleMeterRegistry(),
        )
        repeat(3) {
            assertTrue(limiter.tryAcquire(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, "1.2.3.4").allowed)
        }
        val denied = limiter.tryAcquire(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, "1.2.3.4")
        assertFalse(denied.allowed)
        assertTrue(denied.retryAfter > Duration.ZERO)
    }

    @Test
    fun `buckets are isolated per identity and per policy`() {
        val limiter = Bucket4jRateLimiter(
            RateLimitProperties(
                authLogin = policy(1, Duration.ofMinutes(15)),
                authRegister = policy(1, Duration.ofMinutes(15)),
            ),
            SimpleMeterRegistry(),
        )
        assertTrue(limiter.tryAcquire(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, "a").allowed)
        assertFalse(limiter.tryAcquire(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, "a").allowed)
        // Another identity and another policy each get their own budget.
        assertTrue(limiter.tryAcquire(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, "b").allowed)
        assertTrue(limiter.tryAcquire(Bucket4jRateLimiter.POLICY_AUTH_REGISTER, "a").allowed)
    }

    @Test
    fun `every declared policy resolves to a configured bucket`() {
        // A policy constant with no branch in policyFor() would throw here —
        // this is the net that catches a call-site/config mismatch.
        val limiter = Bucket4jRateLimiter(RateLimitProperties(), SimpleMeterRegistry())
        Bucket4jRateLimiter.ALL_POLICIES.forEach { policy ->
            assertTrue(limiter.tryAcquire(policy, "probe").allowed, "policy $policy denied its first token")
        }
    }

    @Test
    fun `unknown policy fails loudly instead of silently defaulting`() {
        val limiter = Bucket4jRateLimiter(RateLimitProperties(), SimpleMeterRegistry())
        assertFailsWith<IllegalStateException> { limiter.tryAcquire("no-such-policy", "x") }
    }

    @Test
    fun `eviction drops refilled buckets and keeps drained ones`() {
        val limiter = Bucket4jRateLimiter(
            RateLimitProperties(
                // Refills essentially instantly — will be back at full capacity.
                authLogin = policy(2, Duration.ofMillis(1)),
                // 15-minute window — stays drained for the whole test.
                authRegister = policy(2, Duration.ofMinutes(15)),
            ),
            SimpleMeterRegistry(),
        )
        limiter.tryAcquire(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, "fast")
        limiter.tryAcquire(Bucket4jRateLimiter.POLICY_AUTH_REGISTER, "slow")
        limiter.tryAcquire(Bucket4jRateLimiter.POLICY_AUTH_REGISTER, "slow")
        assertEquals(2, limiter.bucketCount())

        Thread.sleep(20) // let the 1ms-period bucket refill to full
        limiter.evictIdleBuckets()

        assertEquals(1, limiter.bucketCount())
        // The drained bucket survived eviction with its state intact — a wrongly
        // evicted bucket would have been recreated full and allowed this.
        assertFalse(limiter.tryAcquire(Bucket4jRateLimiter.POLICY_AUTH_REGISTER, "slow").allowed)
    }
}
