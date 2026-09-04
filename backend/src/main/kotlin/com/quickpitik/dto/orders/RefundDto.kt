package com.quickpitik.dto.orders

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotEmpty
import jakarta.validation.constraints.Size
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// POST /me/orders/{id}/refund — matches website/src/lib/api-orders.ts RefundSubmitArgs.
// Creates one Dispute row per photoId; reason + note are shared across the batch.
data class RefundRequest(
    @field:NotEmpty(message = "photoIds must not be empty")
    val photoIds: List<UUID> = emptyList(),
    @field:NotBlank(message = "reason is required")
    val reason: String = "",
    // Unbounded TEXT in the DB, so cap it at the boundary — the note is copied
    // onto every dispute row in the batch and re-read on each order detail
    // fetch. 500 matches the admin-side reason fields.
    @field:Size(max = 500, message = "note must be 500 characters or fewer")
    val note: String = "",
)

data class RefundResponse(
    val orderId: UUID,
    val disputes: List<RefundDisputeDto>,
)

// status is the FE wire string ("open" | "resolved" | "denied" | "escalated") — keep
// it lowercase to match website/src/lib/admin-disputes.ts DisputeStatus.
data class RefundDisputeDto(
    val id: UUID,
    val photoId: UUID,
    val status: String,
    val openedAt: OffsetDateTime,
)

// Embedded in OrderListItemDto + OrderDetailDto. Carries everything the
// runner-side /orders surface needs to render status chips, the timeline,
// the admin's resolution note, and the cancel button — without a separate
// round-trip per order. resolutionNote is the most-recent
// admin_decision_log.reason for this dispute (resolve / deny / escalate).
data class RunnerDisputeDto(
    val id: UUID,
    val photoId: UUID,
    val reason: String,
    val note: String,
    val status: String,
    val resolution: String?,
    val refundAmount: BigDecimal?,
    val resolutionNote: String?,
    val openedAt: OffsetDateTime,
    val resolvedAt: OffsetDateTime?,
    val withdrawnAt: OffsetDateTime?,
)
