package com.quickpitik.service.ratelimit

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.stereotype.Service
import java.time.Duration

// Default rate limiter — used when `app.rate-limit.enabled=false` (default).
// Lets every request through, returns saturating "remaining". Production flips
// the property to `true` and Bucket4jRateLimiter (below) takes over.
@Service
@ConditionalOnProperty(prefix = "app.rate-limit", name = ["enabled"], havingValue = "false", matchIfMissing = true)
class NoopRateLimiter : RateLimiter {
    override fun tryAcquire(policy: String, key: String): RateLimiter.Decision =
        RateLimiter.Decision(allowed = true, retryAfter = Duration.ZERO, remaining = Long.MAX_VALUE)
}
