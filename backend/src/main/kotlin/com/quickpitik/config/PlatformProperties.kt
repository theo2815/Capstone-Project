package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.math.BigDecimal
import java.time.Duration

@ConfigurationProperties(prefix = "app.platform")
data class PlatformProperties(
    val photoPricePhp: BigDecimal = BigDecimal("125"),
    val platformCutRate: BigDecimal = BigDecimal("0.25"),
    // How long an order's share_token authorizes the three token-gated guest
    // endpoints (status / detail / download-bundle). Measured from order
    // creation. Shorter than the 1-year download grant on purpose: the token
    // is a bearer credential that travels in email, the grant is the actual
    // entitlement and is reachable with a JWT for as long as it lasts.
    val shareTokenTtl: Duration = Duration.ofDays(90),
) {
    val photographerKeepRate: BigDecimal
        get() = BigDecimal.ONE.subtract(platformCutRate)
}
