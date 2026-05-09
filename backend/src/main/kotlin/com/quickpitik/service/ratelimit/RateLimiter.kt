package com.quickpitik.service.ratelimit

import java.time.Duration

interface RateLimiter {
    fun tryAcquire(policy: String, key: String): Decision

    data class Decision(
        val allowed: Boolean,
        val retryAfter: Duration = Duration.ZERO,
        val remaining: Long = 0L,
    )
}
