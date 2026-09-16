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
    // After ANY payout-account change (add/edit/set-primary/delete/QR), block new
    // payout requests for this long — a compromised account can't change the
    // destination and withdraw in the same session. Overridable: PAYOUT_ACCOUNT_COOLDOWN.
    val payoutAccountCooldown: Duration = Duration.ofHours(24),
    // Payouts at or above this amount need a second, DIFFERENT admin to approve
    // before they can be paid. Dormant until >= 2 admin accounts exist (a lone
    // admin must still be able to approve). Overridable: PAYOUT_TWO_PERSON_THRESHOLD_PHP.
    val payoutTwoPersonThresholdPhp: BigDecimal = BigDecimal("10000"),
    // Which disbursement rail moves payout money: "manual" (admin sends by hand
    // and attests via mark-paid — the default) or "paymongo" (create a Transfer
    // via the API; confirmation arrives by webhook). Overridable:
    // PAYOUT_DISBURSEMENT_PROVIDER. Keep "manual" until PayMongo KYB is approved
    // and the adapter is verified in a test-mode account (Phase 11).
    val disbursementProvider: String = "manual",
    // Largest payout an automated (non-manual) transfer may carry. Default is the
    // InstaPay ceiling; raise it only with a verified PESONet rail. Larger cycles
    // are refused with a 409 and settled manually. Overridable: PAYOUT_MAX_PER_TRANSFER_PHP.
    val payoutMaxPerTransferPhp: BigDecimal = BigDecimal("50000"),
    // Step-up re-auth master switch (like AI_API_ENABLED). OFF by default so the
    // shipped web + mobile clients — which don't yet send an X-Step-Up header —
    // keep working. Turn ON only once both clients do the /me/reauth flow; then
    // payout-account changes and admin mark-paid require a fresh password.
    // Overridable: PAYOUT_STEP_UP_ENABLED.
    val payoutStepUpEnabled: Boolean = false,
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
