package com.quickpitik.dto.profile

import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// Mirrors website/src/store/user-media-store.ts SelfieRef.
// `dataUrl` is the wire field name (FE comment notes a future rename to imageUrl
// is deferred — the backend matches the FE contract until that follow-up lands).
data class SelfieRefDto(
    val id: UUID,
    val dataUrl: String,
    val uploadedAt: OffsetDateTime,
    val isPrimary: Boolean,
    val qualityScore: BigDecimal,
)
