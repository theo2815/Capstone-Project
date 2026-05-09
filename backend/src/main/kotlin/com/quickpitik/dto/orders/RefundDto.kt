package com.quickpitik.dto.orders

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotEmpty
import java.time.OffsetDateTime
import java.util.UUID

// POST /me/orders/{id}/refund — matches website/src/lib/api-orders.ts RefundSubmitArgs.
// Creates one Dispute row per photoId; reason + note are shared across the batch.
data class RefundRequest(
    @field:NotEmpty(message = "photoIds must not be empty")
    val photoIds: List<UUID> = emptyList(),
    @field:NotBlank(message = "reason is required")
    val reason: String = "",
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
