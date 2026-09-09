package com.quickpitik.service.disbursement

import java.math.BigDecimal

/**
 * The one seam every payout disbursement goes through, so the mechanism that
 * actually moves money is swappable without touching the lifecycle. Two
 * implementations exist:
 *   - [ManualDisbursementProvider] — the admin sends the money by hand and
 *     attests to it via mark-paid. No external transfer, so no id.
 *   - PayMongoDisbursementProvider (Phase 11) — creates a Transfer via the
 *     PayMongo API; confirmation then arrives via webhook. Disabled by default.
 *
 * The active provider is chosen by `app.platform.disbursement-provider`
 * (default "manual"); [DisbursementService] resolves it.
 */
interface DisbursementProvider {
    /** Config value that selects this provider. */
    val providerName: String

    /**
     * Start moving [request].amountPhp to the snapshotted destination. Returns
     * the provider's transfer id (null for manual — nothing external was
     * created). Must be idempotent on [request].cycleId where the provider
     * supports it: the cycle id is stable, so a retry never double-sends.
     */
    fun initiate(request: DisbursementRequest): DisbursementResult
}

/**
 * The immutable destination snapshot + amount for one cycle. Built from the
 * cycle's account_snapshot_* columns (captured at request time), never the
 * live payout account — so a later account change can't reroute the money.
 */
data class DisbursementRequest(
    val cycleId: String,
    val amountPhp: BigDecimal,
    val method: String, // wire form: gcash | maya | gotyme
    val accountNumber: String,
    val accountName: String,
)

data class DisbursementResult(
    /** Provider transfer id; null for manual (no external transfer exists). */
    val providerTransferId: String?,
)
