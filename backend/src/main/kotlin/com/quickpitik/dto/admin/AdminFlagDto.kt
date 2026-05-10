package com.quickpitik.dto.admin

import jakarta.validation.constraints.Size
import java.util.UUID

// Minimal flag DTO. The FE doesn't yet have a strict shape since
// ADMIN_FLAGS_ENABLED is off by default; we ship a plausible default that
// can be tightened once the queue ingestion path lands.
data class AdminFlagDto(
    val id: UUID,
    val targetKind: String,
    val targetId: UUID,
    val reporterId: UUID?,
    val reason: String,
    val note: String,
    val status: String,
    val resolutionNote: String?,
    val resolvedBy: UUID?,
    val resolvedAt: String?,
    val createdAt: String,
)

data class FlagActionRequest(
    @field:Size(max = 1000)
    val resolutionNote: String? = null,
)
