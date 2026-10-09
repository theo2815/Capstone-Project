package com.quickpitik.dto.admin

import com.quickpitik.entity.AdminActionCommand
import java.time.Instant
import java.util.UUID

data class AdminActionCommandDto(
    val commandId: UUID,
    val state: String,
    val resourceId: UUID?,
    val decisionId: UUID?,
    val messageId: UUID?,
    val completedAt: Instant?,
)

fun AdminActionCommand.toDto() = AdminActionCommandDto(
    commandId = commandId,
    state = state.name,
    resourceId = resultResourceId,
    decisionId = resultDecisionId,
    messageId = resultMessageId,
    completedAt = completedAt,
)
