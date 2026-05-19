package com.quickpitik.dto.runner

import java.time.OffsetDateTime
import java.util.UUID

// Wire shape for /api/v1/me/runner/messages. Mirrors FE RunnerMessage
// (website/src/lib/api-runner-messages.ts). `orderId` deep-links the
// dispute-outcome notifications to /orders?expand={orderId}. `title` is
// non-null only for admin_message; every other kind has title=null and
// the FE derives the label from MESSAGE_KIND_LABEL[kind].
//
// MarkAllReadResponse + MessageRemovedResponse are reused from the
// photographer DTOs (structurally identical, no role-specific fields).
data class RunnerMessageDto(
    val id: UUID,
    /** Lowercase snake_case wire form — see RunnerMessageKind.kt. */
    val kind: String,
    val title: String?,
    val body: String,
    val orderId: UUID?,
    val sourceDecisionId: UUID?,
    val createdAt: OffsetDateTime,
    val readAt: OffsetDateTime?,
)
