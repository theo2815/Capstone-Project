package com.quickpitik.dto.admin

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
import java.util.UUID

// Mirrors website/src/lib/admin-payout-reports.ts PayoutReport.
data class AdminPayoutReportDto(
    val id: UUID,
    val payoutCycleId: String,
    val photographerId: UUID,
    val photographerName: String,
    val handle: String?,
    val reason: String,
    val note: String,
    val status: String,
    val reportedAt: String,
    val acknowledgedAt: String?,
    val acknowledgeReply: String?,
    val resolvedAt: String?,
    val resolutionNote: String?,
)

// PATCH /admin/payouts/reports/{id}/acknowledge — body { reply }.
data class AcknowledgePayoutReportRequest(
    @field:NotBlank
    @field:Size(max = 1000)
    val reply: String,
)

// PATCH /admin/payouts/reports/{id}/resolve — body { resolutionNote }.
data class ResolvePayoutReportRequest(
    @field:NotBlank
    @field:Size(max = 1000)
    val resolutionNote: String,
)
