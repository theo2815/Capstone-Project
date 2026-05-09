package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.math.BigDecimal

@ConfigurationProperties(prefix = "app.platform")
data class PlatformProperties(
    val photoPricePhp: BigDecimal = BigDecimal("125"),
    val platformCutRate: BigDecimal = BigDecimal("0.25"),
) {
    val photographerKeepRate: BigDecimal
        get() = BigDecimal.ONE.subtract(platformCutRate)
}
