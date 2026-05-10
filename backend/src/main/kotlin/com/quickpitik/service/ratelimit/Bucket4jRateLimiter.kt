package com.quickpitik.service.ratelimit

import com.quickpitik.config.RateLimitProperties
import io.github.bucket4j.Bandwidth
import io.github.bucket4j.Bucket
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.stereotype.Service
import java.time.Duration
import java.util.concurrent.ConcurrentHashMap

// In-memory token-bucket. Active when `app.rate-limit.enabled=true`. Buckets
// are keyed by `policy:identityKey` so a single user is rate-limited per
// policy without bleeding into another user's bucket. Multi-instance scaling
// (Redis-backed bucket store) is deferred — see plan's scaling-path notes.
@Service
@ConditionalOnProperty(prefix = "app.rate-limit", name = ["enabled"], havingValue = "true")
class Bucket4jRateLimiter(
    private val properties: RateLimitProperties,
) : RateLimiter {
    private val buckets: ConcurrentHashMap<String, Bucket> = ConcurrentHashMap()

    override fun tryAcquire(policy: String, key: String): RateLimiter.Decision {
        val bucket = buckets.computeIfAbsent("$policy:$key") { createBucket(policy) }
        val probe = bucket.tryConsumeAndReturnRemaining(1)
        return if (probe.isConsumed) {
            RateLimiter.Decision(
                allowed = true,
                retryAfter = Duration.ZERO,
                remaining = probe.remainingTokens,
            )
        } else {
            RateLimiter.Decision(
                allowed = false,
                retryAfter = Duration.ofNanos(probe.nanosToWaitForRefill),
                remaining = probe.remainingTokens,
            )
        }
    }

    private fun createBucket(policy: String): Bucket {
        val (capacity, period) = when (policy) {
            POLICY_PHOTOGRAPHER_UPLOAD ->
                properties.photographerUpload.capacity to properties.photographerUpload.refillPeriod
            POLICY_PUBLIC_GALLERY ->
                properties.publicGallery.capacity to properties.publicGallery.refillPeriod
            else -> 60L to Duration.ofMinutes(1)
        }
        return Bucket.builder()
            .addLimit(
                Bandwidth.builder()
                    .capacity(capacity)
                    .refillGreedy(capacity, period)
                    .build(),
            )
            .build()
    }

    companion object {
        const val POLICY_PHOTOGRAPHER_UPLOAD = "photographer-upload"
        const val POLICY_PUBLIC_GALLERY = "public-gallery"
    }
}
