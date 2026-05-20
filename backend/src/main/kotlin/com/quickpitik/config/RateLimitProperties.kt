package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

@ConfigurationProperties(prefix = "app.rate-limit")
data class RateLimitProperties(
    val enabled: Boolean = false,
    val photographerUpload: Policy = Policy(capacity = 600, refillPeriod = Duration.ofMinutes(1)),
    val publicGallery: Policy = Policy(capacity = 60, refillPeriod = Duration.ofMinutes(1)),
    // Pre-auth endpoints — keyed by client IP since there's no userId yet.
    // Per-minute caps are sized for normal humans plus a small retry cushion;
    // anything beyond is credential-stuffing / email-bombing / token-brute.
    val authLogin: Policy = Policy(capacity = 10, refillPeriod = Duration.ofMinutes(1)),
    val authRegister: Policy = Policy(capacity = 5, refillPeriod = Duration.ofMinutes(1)),
    val authForgotPassword: Policy = Policy(capacity = 3, refillPeriod = Duration.ofMinutes(1)),
    val authResetPassword: Policy = Policy(capacity = 5, refillPeriod = Duration.ofMinutes(1)),
) {
    data class Policy(
        val capacity: Long,
        val refillPeriod: Duration,
    )
}
