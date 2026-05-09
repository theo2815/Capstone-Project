package com.quickpitik.service.ratelimit

import org.springframework.stereotype.Service
import java.time.Duration

@Service
class NoopRateLimiter : RateLimiter {
    override fun tryAcquire(policy: String, key: String): RateLimiter.Decision =
        RateLimiter.Decision(allowed = true, retryAfter = Duration.ZERO, remaining = Long.MAX_VALUE)
}
