package com.quickpitik.dto.admin

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// One entry per admin action on this dispute. Sourced from admin_decision_log
// rows where target_dispute_id = X. `resolution` + `refundAmount` are
// flattened out of the `meta` JSONB (decision="resolved" only) so the FE
// can render them without parsing JSON.
data class DisputeActivityEntry(
    val id: UUID,
    val decidedAt: OffsetDateTime,
    val decision: String,
    val resolution: String?,
    val refundAmount: BigDecimal?,
    val reason: String?,
)

// Mirrors website/src/lib/admin-disputes.ts Dispute. orderSnapshot +
// photoSnapshot are computed at read time from the live order + photo rows
// — Phase G ships the ledger-of-truth view; future archival snapshots can
// freeze them at submit time if the requirement firms up.
data class DisputeOrderSnapshotDto(
    val total: BigDecimal,
    val paymentMethod: String,
    val paidAt: String?,
)

data class DisputePhotoSnapshotDto(
    val alt: String,
    val kmMark: BigDecimal?,
    val bib: String?,
    val thumbnailUrl: String?,
)

data class AdminDisputeDto(
    val id: UUID,
    val orderId: UUID,
    val photoId: UUID,
    val eventId: UUID,
    val eventName: String?,
    val runnerHandle: String,
    val photographerHandle: String,
    val reason: String,
    val note: String,
    val status: String,
    val reportedAt: String,
    val resolvedAt: String?,
    val refundAmount: BigDecimal?,
    val resolution: String?,
    val orderSnapshot: DisputeOrderSnapshotDto,
    val photoSnapshot: DisputePhotoSnapshotDto,
    // Every admin action on this dispute, newest first. Sourced from
    // admin_decision_log so past-session decisions persist. Empty for
    // open disputes (no admin action yet) or disputes closed before the
    // decision log went live.
    val activity: List<DisputeActivityEntry> = emptyList(),
)

// POST /admin/disputes/{id}/resolve — body { resolution, refundAmount?, reason }.
// resolution is one of refund_full | refund_partial | deny.
data class ResolveDisputeRequest(
    @field:NotBlank
    val resolution: String,
    val refundAmount: BigDecimal? = null,
    @field:Size(max = 500)
    val reason: String? = null,
)

// POST /admin/disputes/{id}/deny — body { reason }.
//
// `reason` is shown to the runner: verbatim in the inbox message body, and as
// `RunnerDisputeDto.resolutionNote` in the /orders refund timeline. Write it
// for the customer. See the KDoc on AdminDisputeService.deny.
data class DenyDisputeRequest(
    @field:Size(max = 500)
    val reason: String? = null,
)

// POST /admin/disputes/{id}/escalate — body { reason }.
data class EscalateDisputeRequest(
    @field:Size(max = 500)
    val reason: String? = null,
)
