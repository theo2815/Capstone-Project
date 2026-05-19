package com.quickpitik.dto.admin

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotEmpty
import jakarta.validation.constraints.Size
import java.math.BigDecimal
import java.util.UUID

// Mirrors website/src/lib/admin-payouts.ts PayoutAccountSnapshot. The QR
// dataUrl is a presigned GET URL when there's an S3 key and null otherwise —
// the FE accepts either a base64 data URL or an http(s) URL on <img src>.
data class AdminPayoutAccountSnapshotDto(
    val method: String,
    val accountNumber: String,
    val accountName: String,
    val qr: AdminPayoutQrDto?,
)

data class AdminPayoutQrDto(
    val dataUrl: String,
    val uploadedAt: String,
)

// Mirrors AdminPayoutCycle. The DB enum has scheduled / pending / paid / held;
// the FE filter uses pending_review / approved / held / paid. We translate
// at the boundary: DB pending → wire pending_review; once the admin approves
// a row it stays at DB pending until mark-paid (held is held); approved
// state is internal — service layer decides the wire form by inspecting
// settledAt + paymentReference + holdReason.
data class AdminPayoutCycleDto(
    val id: String,
    val photographerId: UUID,
    val photographerName: String,
    val brandName: String?,
    val handle: String?,
    val weekOf: String,
    val amount: BigDecimal,
    val itemCount: Int,
    val method: String,
    val status: String,
    val submittedAt: String,
    val reviewedAt: String?,
    val paidAt: String?,
    val paymentReference: String?,
    val holdReason: String?,
    val payoutAccount: AdminPayoutAccountSnapshotDto,
)

// POST /admin/payouts/{id}/hold — body { reason }.
data class HoldPayoutRequest(
    @field:NotBlank
    @field:Size(max = 255)
    val reason: String,
)

// POST /admin/payouts/{id}/mark-paid — body { paymentReference }.
data class MarkPayoutPaidRequest(
    @field:NotBlank
    @field:Size(max = 100)
    val paymentReference: String,
)

// POST /admin/payouts/bulk — body { ids, action, reason? }.
// Idempotency-Key header opt-in: if present, replays return the same group_id.
data class BulkPayoutsRequest(
    @field:NotEmpty
    val ids: List<String>,
    @field:NotBlank
    val action: String,
    val reason: String? = null,
)

// Mirrors api-admin.ts BulkPayoutResult.
data class BulkPayoutItemResultDto(
    val id: String,
    val ok: Boolean,
    val error: String? = null,
)

data class BulkPayoutResultDto(
    val groupId: UUID,
    val results: List<BulkPayoutItemResultDto>,
)

// POST /admin/payouts/generate?weekOf=YYYY-MM-DD. Idempotent on (weekOf,
// photographer): re-running for the same week updates pending-review rows in
// place, leaves admin-decided rows (paid / held / already-approved) untouched.
data class GenerateCyclesResultDto(
    val weekOf: String,
    val created: Int,
    val updated: Int,
    // Cycles that already had an admin decision applied (paid/held/approved)
    // and were left alone so the generator doesn't clobber human judgment.
    val skipped: Int,
    val totalAmount: BigDecimal,
)
