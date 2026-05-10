package com.quickpitik.dto.admin

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
import java.time.OffsetDateTime
import java.util.UUID

// Mirrors website/src/lib/admin-user-registry.ts PhotographerSettingsSnapshot.
data class PhotographerSettingsSnapshotDto(
    val hasCover: Boolean,
    val hasBrandName: Boolean,
    val hasWatermark: Boolean,
    val hasHandle: Boolean,
    val hasRegion: Boolean,
    val socialCount: Int,
    val payoutCount: Int,
)

// Mirrors AdminUserRow. `verificationStatus` is lowercase wire form
// (incomplete | pending | approved | rejected) for photographers; runners
// always serialize as "approved" since they have no settings to gate on.
// `region` is the human-readable "Cebu · Central Visayas" — derived from
// region/province codes via RegionsService.
data class AdminUserRowDto(
    val userId: UUID,
    val role: String,
    val email: String,
    val name: String,
    val brandName: String?,
    val handle: String?,
    val region: String?,
    val city: String,
    val createdAt: String,
    val verificationStatus: String,
    val suspendedAt: String?,
    val suspensionReason: String?,
    val settingsSnapshot: PhotographerSettingsSnapshotDto?,
)

// Mirrors website/src/store/admin-user-store.ts DecisionLogEntry.
data class DecisionLogEntryDto(
    val userId: UUID,
    val decision: String,
    val reason: String?,
    val decidedAt: OffsetDateTime,
    val meta: Map<String, Any?>?,
)

// Mirrors api-admin.ts AdminUserDetail = AdminUserRow + decisionLog[].
data class AdminUserDetailDto(
    val userId: UUID,
    val role: String,
    val email: String,
    val name: String,
    val brandName: String?,
    val handle: String?,
    val region: String?,
    val city: String,
    val createdAt: String,
    val verificationStatus: String,
    val suspendedAt: String?,
    val suspensionReason: String?,
    val settingsSnapshot: PhotographerSettingsSnapshotDto?,
    val decisionLog: List<DecisionLogEntryDto>,
)

// POST /admin/users/{id}/reject — body { reason }.
data class RejectUserRequest(
    @field:NotBlank
    @field:Size(max = 500)
    val reason: String,
)

// POST /admin/users/{id}/suspend — body { reason }.
data class SuspendUserRequest(
    @field:NotBlank
    @field:Size(max = 500)
    val reason: String,
)

// PATCH /admin/users/{id} — body { handle?, brandName? }. Both optional;
// service-side validates non-blank values against the same handle regex +
// reserved-handle gate the photographer's own PUT uses.
data class ForceEditUserPatchRequest(
    val handle: String? = null,
    val brandName: String? = null,
)
