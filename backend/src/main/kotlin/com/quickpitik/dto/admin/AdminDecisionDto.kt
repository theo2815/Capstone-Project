package com.quickpitik.dto.admin

import io.swagger.v3.oas.annotations.media.Schema
import java.time.OffsetDateTime
import java.util.UUID

// One row of GET /admin/decisions (the overview's "recent decisions" list).
data class AdminDecisionDto(
    val id: UUID,
    val decision: String,
    val reason: String?,
    val decidedAt: OffsetDateTime,
    @field:Schema(description = "user | payout | dispute | event | removal; null when the row points at nothing")
    val targetType: String?,
    val targetId: String?,
    val adminId: UUID,
    @field:Schema(description = "Admin display name, else e-mail; null when the admin account no longer exists")
    val adminName: String?,
)
