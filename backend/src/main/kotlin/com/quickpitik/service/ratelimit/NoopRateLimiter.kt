package com.quickpitik.service.ratelimit

import org.slf4j.LoggerFactory
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.core.env.Environment
import org.springframework.stereotype.Service
import java.time.Duration

// Default rate limiter — used when `app.rate-limit.enabled=false` (default).
// Lets every request through, returns saturating "remaining". Production flips
// the property to `true` and Bucket4jRateLimiter (below) takes over.
@Service
@ConditionalOnProperty(prefix = "app.rate-limit", name = ["enabled"], havingValue = "false", matchIfMissing = true)
class NoopRateLimiter(
    private val environment: Environment,
) : RateLimiter {
    private val log = LoggerFactory.getLogger(javaClass)

    init {
        log.info("NoopRateLimiter active — rate limiting is OFF (app.rate-limit.enabled=false)")
        // RATE_LIMIT_ENABLED defaults to false, so forgetting to set it in prod
        // silently unprotects /auth/login, /auth/forgot-password and photo
        // upload. Mirrors the L-2 warning in LocalFsStorageService: empty or
        // "default" profiles are the implicit-dev case.
        val activeProfiles = environment.activeProfiles.toList()
        val isDevLike = activeProfiles.isEmpty() ||
            activeProfiles.any { it.equals("dev", ignoreCase = true) || it.equals("default", ignoreCase = true) }
        if (!isDevLike) {
            log.warn(
                "Rate limiting is DISABLED with non-dev profile {}. " +
                    "Login, registration, forgot-password and photo upload have no throttle — " +
                    "credential stuffing and email bombing are unbounded. " +
                    "Set RATE_LIMIT_ENABLED=true in production.",
                activeProfiles,
            )
        }
    }

    override fun tryAcquire(policy: String, key: String): RateLimiter.Decision =
        RateLimiter.Decision(allowed = true, retryAfter = Duration.ZERO, remaining = Long.MAX_VALUE)
}
