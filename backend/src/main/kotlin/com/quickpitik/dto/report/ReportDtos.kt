package com.quickpitik.dto.report

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size

// POST /me/reports/events/{eventId} and /me/reports/photographers/{handle} (AA-R9).
data class ReportRequest(
    @field:NotBlank(message = "note is required")
    @field:Size(min = 10, max = 500, message = "note must be 10-500 characters")
    val note: String,
)

// alreadyReported: this account already has an open report on the target; nothing new was written.
data class ReportResultDto(
    val reported: Boolean = true,
    val alreadyReported: Boolean,
)
