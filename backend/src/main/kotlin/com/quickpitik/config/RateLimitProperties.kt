package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

@ConfigurationProperties(prefix = "app.rate-limit")
data class RateLimitProperties(
    val enabled: Boolean = false,
    val photographerUpload: Policy = Policy(capacity = 600, refillPeriod = Duration.ofMinutes(1)),
    val publicGallery: Policy = Policy(capacity = 60, refillPeriod = Duration.ofMinutes(1)),
) {
    data class Policy(
        val capacity: Long,
        val refillPeriod: Duration,
    )
}
