package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

@ConfigurationProperties(prefix = "app.admin-action")
data class AdminActionProperties(val lease: Duration = Duration.ofMinutes(5)) {
    init { require(lease > Duration.ZERO) { "Admin command lease must be positive" } }
}
