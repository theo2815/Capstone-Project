package com.quickpitik.dto.admin

import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Size
import java.util.UUID

// PE-2.4: one photo a relevance flag's batch hide recorded. A restored row stays in the list, with restoredAt set.
data class HiddenPhotoDto(
    val photoId: UUID,
    val eventId: UUID,
    val thumbnailUrl: String?,
    val hiddenAt: String,
    val restoredAt: String?,
)

// PE-2.4: POST /api/v1/admin/flags/{flagId}/hide-batch.
data class HideBatchRequest(
    @field:NotNull
    val eventId: UUID? = null,
    @field:Size(max = 1000)
    val resolutionNote: String? = null,
)
