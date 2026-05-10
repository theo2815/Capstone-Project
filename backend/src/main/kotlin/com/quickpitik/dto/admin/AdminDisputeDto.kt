package com.quickpitik.dto.admin

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
import java.math.BigDecimal
import java.util.UUID

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
data class DenyDisputeRequest(
    @field:Size(max = 500)
    val reason: String? = null,
)

// POST /admin/disputes/{id}/escalate — body { reason }.
data class EscalateDisputeRequest(
    @field:Size(max = 500)
    val reason: String? = null,
)
