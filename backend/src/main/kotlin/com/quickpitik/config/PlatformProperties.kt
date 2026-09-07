package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.math.BigDecimal
import java.time.Duration

@ConfigurationProperties(prefix = "app.platform")
data class PlatformProperties(
    val photoPricePhp: BigDecimal = BigDecimal("125"),
    val platformCutRate: BigDecimal = BigDecimal("0.25"),
    // Largest photographer coupon, as a whole percentage of the photographer's
    // share. Whatever the value, the platform cut on a sale never moves.
    val couponMaxPercent: Int = 50,
    // Earnings become withdrawable only after this window elapses from the sale,
    // so a refund/dispute in the meantime is absorbed before any payout. Align
    // to the refund/dispute-eligibility window. Overridable: PAYOUT_CLEARING_WINDOW.
    val payoutClearingWindow: Duration = Duration.ofDays(7),
    // Upper bound for emailed bundle capabilities and migrated legacy tokens.
    // The user-facing "works for 90 days" copy is hardcoded to this default —
    // if you change it, update the receipt email (OrderReceiptEmailService) and
    // the website /orders/return page copy too.
    val shareTokenTtl: Duration = Duration.ofDays(90),
    // Long enough for a 30-minute QRPH payment plus webhook/polling delay.
    val orderReturnTtl: Duration = Duration.ofMinutes(35),
    // Separate HMAC key for purpose-bound order return/download links.
    val orderCapabilitySecret: String =
        "dev-only-order-capability-secret-DO-NOT-USE-IN-PRODUCTION-replace-with-32-byte-random",
) {
    val photographerKeepRate: BigDecimal
        get() = BigDecimal.ONE.subtract(platformCutRate)
}
