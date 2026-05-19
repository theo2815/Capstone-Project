package com.quickpitik.dto.earnings

import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutReport
import com.quickpitik.entity.Transaction
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

// ─── Overview ────────────────────────────────────────────────────────────
// Mirrors website/src/lib/photographer-mock.ts PhotographerEarnings + WeeklyRevenuePoint.
data class WeeklyRevenuePointDto(
    val weekOf: LocalDate,
    val amount: BigDecimal,
)

data class EarningsOverviewDto(
    val lifetimeKept: BigDecimal,
    val thisWeek: BigDecimal,
    val thisMonth: BigDecimal,
    val payoutPending: BigDecimal,
    // Null when no upcoming cycle exists — FE renders an em-dash placeholder.
    val payoutScheduledFor: LocalDate?,
    val weeklySeries: List<WeeklyRevenuePointDto>,
    val thisWeekSold: Long,
    val thisMonthSold: Long,
)

// ─── Per-event earnings ──────────────────────────────────────────────────
// Mirrors website/src/lib/api-photographer-earnings.ts PerEventEarning. eventDate
// is ISO date-only (YYYY-MM-DD).
data class PerEventEarningDto(
    val eventId: String,
    val eventName: String,
    val eventDate: LocalDate?,
    val photoCount: Int,
    val salesCount: Int,
    val revenueKept: BigDecimal,
)

// ─── Cycles ──────────────────────────────────────────────────────────────
// Mirrors website/src/lib/photographer-mock.ts PhotographerPayout. id is the
// `PAY-{YYYY}W{NN}-{HANDLE}` string per Q-A1; status + method are lowercase
// wire form pulled from the entity wire columns.
data class PhotographerPayoutDto(
    val id: String,
    val weekOf: LocalDate,
    val amount: BigDecimal,
    val status: String,
    val settledAt: OffsetDateTime?,
    val method: String,
    val reference: String?,
    // Surfaced for the photographer's "request held" hero so they can read
    // the admin's reason inline (also goes to their inbox). Null in any
    // non-held state.
    val holdReason: String?,
)

fun PayoutCycle.toDto(): PhotographerPayoutDto = PhotographerPayoutDto(
    id = id,
    weekOf = weekOf,
    amount = amountPhp,
    status = statusWire,
    settledAt = settledAt,
    method = methodWire,
    reference = paymentReference,
    holdReason = holdReason,
)

// ─── Reports ─────────────────────────────────────────────────────────────
// Mirrors website/src/lib/admin-payout-reports.ts PayoutReport. The /me/payouts
// reports endpoint is photographer-scoped so photographerId/Name/handle can be
// derived once per request from the requesting principal.
data class PayoutReportDto(
    val id: String,
    val payoutCycleId: String,
    val photographerId: String,
    val photographerName: String,
    val handle: String?,
    val reason: String,
    val note: String,
    val status: String,
    val reportedAt: OffsetDateTime,
    val acknowledgedAt: OffsetDateTime?,
    val acknowledgeReply: String?,
    val resolvedAt: OffsetDateTime?,
    val resolutionNote: String?,
)

fun PayoutReport.toDto(
    photographerName: String,
    handle: String?,
): PayoutReportDto = PayoutReportDto(
    id = id.toString(),
    payoutCycleId = payoutId,
    photographerId = photographerId.toString(),
    photographerName = photographerName,
    handle = handle,
    reason = reasonWire,
    note = note,
    status = statusWire,
    reportedAt = openedAt,
    acknowledgedAt = acknowledgedAt,
    acknowledgeReply = acknowledgeReply,
    resolvedAt = resolvedAt,
    resolutionNote = resolutionNote,
)

data class SubmitPayoutReportRequest(
    @field:NotBlank(message = "reason is required")
    val reason: String,
    // FE may send null, empty string, or a short note. 2000-char cap mirrors
    // the bio limit on settings — same rule, same field type.
    @field:Size(max = 2000, message = "note must be 0-2000 chars")
    val note: String? = null,
)

// ─── Photographer-initiated payout request ──────────────────────────────
// GET /me/photographer/payouts/balance drives the request-payout hero on
// /dashboard/billing. openRequest is the cycle currently in pending_review /
// approved / held — null when the photographer has none, in which case the
// hero shows the request CTA gated on unpaidBalance >= minimum.
data class PayoutBalanceDto(
    val unpaidBalance: BigDecimal,
    val minimum: BigDecimal,
    val hasOpenRequest: Boolean,
    val openRequest: PhotographerPayoutDto?,
)

// ─── Transactions ledger ─────────────────────────────────────────────────
// Mirrors website/src/lib/photographer-mock.ts PhotographerTransaction +
// website/src/lib/api-photographer-earnings.ts TransactionsResponse. monthTotals
// is computed server-side over ALL transactions for the photographer (not just
// the page) so the sticky-header math stays correct as the user scrolls.
data class PhotographerTransactionDto(
    val id: String,
    val paidAt: OffsetDateTime,
    val eventId: String,
    // eventName + eventSlug snapshot the originating event so the billing
    // ledger can render the title (and deep-link) without the FE keeping a
    // parallel events lookup. Null means the event was hard-deleted — FE
    // falls back to "Event archived".
    val eventName: String?,
    val eventSlug: String?,
    val photoId: String,
    val buyer: String,
    val amountKept: BigDecimal,
)

fun Transaction.toDto(
    buyerDisplay: String,
    eventName: String?,
    eventSlug: String?,
): PhotographerTransactionDto =
    PhotographerTransactionDto(
        id = id.toString(),
        paidAt = paidAt,
        eventId = eventId.toString(),
        eventName = eventName,
        eventSlug = eventSlug,
        photoId = photoId.toString(),
        buyer = buyerDisplay,
        amountKept = amountKeptPhp,
    )

data class TransactionsLedgerResponse(
    val items: List<PhotographerTransactionDto>,
    val total: Long,
    val offset: Int,
    val limit: Int,
    val monthTotals: Map<String, BigDecimal>,
)

// Internal: converts a User.name to "Aira S." privacy-safe display per FE
// PhotographerTransaction.buyer convention. Single name → return as-is. Empty
// → empty string (the buyer_id was nulled by ON DELETE SET NULL).
fun privacyBuyerDisplay(name: String?): String {
    if (name.isNullOrBlank()) return ""
    val parts = name.trim().split(Regex("\\s+")).filter { it.isNotBlank() }
    if (parts.isEmpty()) return ""
    val first = parts.first()
    if (parts.size == 1) return first
    val lastInitial = parts.last().firstOrNull()?.uppercase() ?: return first
    return "$first $lastInitial."
}

// Internal: builds the legacy week-based cycle id `PAY-{YYYY}W{NN}-{HANDLE}` —
// kept for the smoke-test seed helper (one cycle per week per photographer).
// The live request-based flow uses buildSequencedCycleId instead so multiple
// requests in the same ISO week don't collide on the unique PK.
fun buildCycleId(weekOf: LocalDate, handle: String?): String {
    val isoYear = weekOf.get(java.time.temporal.IsoFields.WEEK_BASED_YEAR)
    val isoWeek = weekOf.get(java.time.temporal.IsoFields.WEEK_OF_WEEK_BASED_YEAR)
    val handlePart = handle?.uppercase()?.replace(Regex("[^A-Z0-9]"), "")
        ?.takeIf { it.isNotEmpty() }
        ?: "UNKNOWN"
    return "PAY-${isoYear}W${"%02d".format(isoWeek)}-$handlePart"
}

// Photographer-initiated payouts can fire multiple times per ISO week, so the
// week-based id collides on the PK. Sequence id is monotonic per photographer:
// PAY-JUANDC-001, PAY-JUANDC-002, etc. Handle sanitised the same way; falls
// back to a UUID prefix when handle is missing so the id is still stable.
fun buildSequencedCycleId(handle: String?, photographerId: UUID, sequence: Long): String {
    val handlePart = handle?.uppercase()?.replace(Regex("[^A-Z0-9]"), "")
        ?.takeIf { it.isNotEmpty() }
        ?: photographerId.toString().take(8).uppercase()
    return "PAY-$handlePart-${"%03d".format(sequence)}"
}

// Used by the /me/photographer/payouts/{id} test-seed endpoint signature so a
// caller can supply explicit fields. Intentionally not exported to the FE
// since the FE does not generate cycles (admin or cron does).
data class CreatePayoutCycleRequest(
    val weekOf: LocalDate,
    val amountPhp: BigDecimal,
    val itemCount: Int,
    val method: String,
    val status: String? = null,
    val settledAt: OffsetDateTime? = null,
    val paymentReference: String? = null,
    val photographerId: UUID? = null,
)
