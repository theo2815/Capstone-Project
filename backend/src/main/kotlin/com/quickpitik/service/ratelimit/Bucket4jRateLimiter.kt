package com.quickpitik.service.ratelimit

import com.quickpitik.config.RateLimitProperties
import io.github.bucket4j.Bandwidth
import io.github.bucket4j.Bucket
import io.micrometer.core.instrument.MeterRegistry
import org.slf4j.LoggerFactory
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.scheduling.annotation.Scheduled
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
    private val meterRegistry: MeterRegistry,
) : RateLimiter {
    private val log = LoggerFactory.getLogger(javaClass)
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
            meterRegistry.counter("qp.ratelimit.denied", "policy", policy).increment()
            RateLimiter.Decision(
                allowed = false,
                retryAfter = Duration.ofNanos(probe.nanosToWaitForRefill),
                remaining = probe.remainingTokens,
            )
        }
    }

    // Buckets are only ever inserted (computeIfAbsent), so without a sweep the
    // map grows by one entry per distinct key forever. A bucket refilled back
    // to full capacity holds no state a fresh bucket wouldn't — drop it.
    // ponytail: tiny check-then-remove race can hand one caller a free token;
    // harmless for throttling, not worth CAS gymnastics.
    @Scheduled(fixedDelayString = "\${app.rate-limit.evict-interval-ms:3600000}")
    fun evictIdleBuckets() {
        var removed = 0
        buckets.entries.removeIf { entry ->
            val full = entry.value.availableTokens >= policyFor(entry.key.substringBefore(':')).capacity
            if (full) removed++
            full
        }
        if (removed > 0) log.debug("Evicted {} idle rate-limit bucket(s); {} remain", removed, buckets.size)
    }

    // Test-only visibility (same pattern as ResendClient.isRetryable): the map
    // is the eviction sweep's observable state.
    internal fun bucketCount(): Int = buckets.size

    private fun createBucket(policy: String): Bucket {
        val config = policyFor(policy)
        return Bucket.builder()
            .addLimit(
                Bandwidth.builder()
                    .capacity(config.capacity)
                    .refillGreedy(config.capacity, config.refillPeriod)
                    .build(),
            )
            .build()
    }

    private fun policyFor(policy: String): RateLimitProperties.Policy = when (policy) {
        POLICY_PHOTOGRAPHER_UPLOAD -> properties.photographerUpload
        POLICY_PUBLIC_GALLERY -> properties.publicGallery
        POLICY_PHOTO_SEARCH -> properties.photoSearch
        POLICY_AUTH_LOGIN -> properties.authLogin
        POLICY_AUTH_REGISTER -> properties.authRegister
        POLICY_AUTH_FORGOT_PASSWORD -> properties.authForgotPassword
        POLICY_AUTH_VERIFY_RESET_OTP -> properties.authVerifyResetOtp
        POLICY_AUTH_RESET_PASSWORD -> properties.authResetPassword
        POLICY_ORDER_CREATE -> properties.orderCreate
        POLICY_BUNDLE_DOWNLOAD -> properties.bundleDownload
        POLICY_MEDIA_UPLOAD -> properties.mediaUpload
        // Policies are compile-time constants — an unknown string is a call-site
        // typo, and silently defaulting (the old 60/min fallback) would ship the
        // wrong limit. Die loudly so a test catches it.
        else -> error("Unknown rate-limit policy '$policy' — add it to RateLimitProperties and policyFor()")
    }

    companion object {
        const val POLICY_PHOTOGRAPHER_UPLOAD = "photographer-upload"
        const val POLICY_PUBLIC_GALLERY = "public-gallery"
        const val POLICY_PHOTO_SEARCH = "photo-search"
        const val POLICY_AUTH_LOGIN = "auth-login"
        const val POLICY_AUTH_REGISTER = "auth-register"
        const val POLICY_AUTH_FORGOT_PASSWORD = "auth-forgot-password"
        const val POLICY_AUTH_VERIFY_RESET_OTP = "auth-verify-reset-otp"
        const val POLICY_AUTH_RESET_PASSWORD = "auth-reset-password"
        const val POLICY_ORDER_CREATE = "order-create"
        const val POLICY_BUNDLE_DOWNLOAD = "bundle-download"
        const val POLICY_MEDIA_UPLOAD = "media-upload"

        // Every routable policy, for the resolution test — a constant missing
        // here escapes that net, so keep it in sync with the list above.
        val ALL_POLICIES: List<String> = listOf(
            POLICY_PHOTOGRAPHER_UPLOAD,
            POLICY_PUBLIC_GALLERY,
            POLICY_PHOTO_SEARCH,
            POLICY_AUTH_LOGIN,
            POLICY_AUTH_REGISTER,
            POLICY_AUTH_FORGOT_PASSWORD,
            POLICY_AUTH_VERIFY_RESET_OTP,
            POLICY_AUTH_RESET_PASSWORD,
            POLICY_ORDER_CREATE,
            POLICY_BUNDLE_DOWNLOAD,
            POLICY_MEDIA_UPLOAD,
        )
    }
}
