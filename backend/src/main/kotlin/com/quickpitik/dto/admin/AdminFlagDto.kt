package com.quickpitik.dto.admin

import jakarta.validation.constraints.Size
import java.math.BigDecimal
import java.util.UUID

data class FlagPhotoSnapshotDto(
    val alt: String = "",
    val kmMark: BigDecimal? = null,
    val bib: String? = null,
    val thumbnailUrl: String? = null,
)

data class AdminFlagDto(
    val id: UUID,
    val photoId: UUID?,
    val eventId: UUID?,
    val eventName: String?,
    val photographerHandle: String,
    val photographerName: String?,
    val reportedBy: String,
    val reason: String,
    val note: String,
    val status: String,
    val reportedAt: String,
    val reviewedAt: String?,
    val reviewedBy: String?,
    val reviewerNote: String?,
    val photoSnapshot: FlagPhotoSnapshotDto,
    val targetKind: String,
    val targetId: UUID,
    val reporterId: UUID?,
)

data class FlagActionRequest(
    @field:Size(max = 1000)
    val resolutionNote: String? = null,
)
