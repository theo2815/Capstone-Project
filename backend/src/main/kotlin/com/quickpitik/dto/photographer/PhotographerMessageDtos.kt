package com.quickpitik.dto.photographer

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
import java.time.OffsetDateTime
import java.util.UUID

// Wire shape for /api/v1/me/photographer/messages. Mirrors the FE
// PhotographerMessage type (website/src/lib/photographer-messages.ts).
// `title` is non-null only for admin_message (admin-supplied subject);
// every other kind has title=null and the FE derives the label from
// MESSAGE_KIND_LABEL[kind] so admin actions don't have to invent titles.
data class PhotographerMessageDto(
    val id: UUID,
    /** Lowercase snake_case wire form — see PhotographerMessageKind.kt. */
    val kind: String,
    val title: String?,
    val body: String,
    val sourceDecisionId: UUID?,
    val createdAt: OffsetDateTime,
    val readAt: OffsetDateTime?,
)

// POST /api/v1/admin/users/{id}/message — admin → photographer DM. Subject
// becomes message.title; body is the long-form message. Both required.
data class AdminUserMessageRequest(
    @field:NotBlank
    @field:Size(max = 200)
    val subject: String,

    @field:NotBlank
    @field:Size(max = 5000)
    val body: String,
)

// PATCH /api/v1/me/photographer/messages/read-all + DELETE responses use
// a thin shape so the FE can update the cache deterministically.
data class MarkAllReadResponse(val markedRead: Int)

data class MessageRemovedResponse(val removed: Boolean)
