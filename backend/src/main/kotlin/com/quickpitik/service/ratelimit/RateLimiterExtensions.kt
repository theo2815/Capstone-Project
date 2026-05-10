package com.quickpitik.service.ratelimit

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ApiException
import org.springframework.http.HttpStatus

// Throw-on-deny helper so call sites stay one-liners. Picks up retry-after
// from the bucket's nanos-to-refill so the FE can hold the next attempt.
fun RateLimiter.acquireOrThrow(policy: String, key: String) {
    val decision = tryAcquire(policy, key)
    if (decision.allowed) return
    val retrySeconds = (decision.retryAfter.toSeconds() + 1).coerceAtLeast(1L)
    throw ApiException(
        status = HttpStatus.TOO_MANY_REQUESTS,
        code = ErrorCodes.RATE_LIMITED,
        message = "Too many requests. Please slow down and try again.",
        retryAfterSeconds = retrySeconds,
    )
}
