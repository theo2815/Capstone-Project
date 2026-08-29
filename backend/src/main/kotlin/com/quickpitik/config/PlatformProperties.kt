package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.math.BigDecimal
import java.time.Duration

@ConfigurationProperties(prefix = "app.platform")
data class PlatformProperties(
    val photoPricePhp: BigDecimal = BigDecimal("125"),
    val platformCutRate: BigDecimal = BigDecimal("0.25"),
    // Upper bound for emailed bundle capabilities and migrated legacy tokens.
    val shareTokenTtl: Duration = Duration.ofDays(90),
    // Separate HMAC key for purpose-bound order return/download links.
    val orderCapabilitySecret: String =
        "dev-only-order-capability-secret-DO-NOT-USE-IN-PRODUCTION-replace-with-32-byte-random",
) {
    val photographerKeepRate: BigDecimal
        get() = BigDecimal.ONE.subtract(platformCutRate)
}
