package com.quickpitik.dto.removal

import jakarta.validation.constraints.AssertTrue
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Size
import java.time.Instant
import java.util.UUID

// N2: the runner picks one reason; admins see it, photographers never do.
enum class RunnerRemovalReason(val label: String) {
    PRIVACY("I don't want photos of me online"),
    SAFETY("Safety or security concern"),
    NO_CONSENT("I didn't agree to be photographed"),
    OTHER("Something else"),
}

data class RunnerRemovalReasonDto(val code: String, val label: String)

data class RunnerRemovalEligibilityDto(
    val eligible: Boolean,
    val blockedReason: String?,
    val attemptsLeftToday: Int,
    val emailVerified: Boolean,
    val openRequestId: UUID?,
)

// No photo ids and no image: the server searches with the live check's own reference image.
data class SubmitRunnerRemovalCommand(
    @field:NotBlank val sessionId: String?,
    @field:NotNull val eventId: UUID?,
    @field:NotBlank val reasonCode: String?,
    @field:Size(max = 500) val details: String?,
)

data class RunnerSubmitResultDto(val requestId: UUID, val status: String, val hiddenCount: Int, val suggestedCount: Int)

// RD-5: status, counts and a timeline only: no photo ids, scores, owners, evidence, contact or reasons.
data class RunnerRemovalCountsDto(val hidden: Int = 0, val suggested: Int = 0, val removed: Int = 0, val kept: Int = 0)
data class RunnerRemovalRequestDto(
    val id: UUID, val eventId: UUID, val eventName: String, val status: String, val createdAt: Instant, val updatedAt: Instant,
    val version: Long, val canWithdraw: Boolean, val counts: RunnerRemovalCountsDto,
)
data class RunnerTimelineStepDto(val at: Instant, val step: String)
// Withdraw receipts store and replay this DTO: a field added later needs a default.
data class RunnerRemovalRequestDetailDto(
    val id: UUID, val eventId: UUID, val eventName: String, val status: String, val createdAt: Instant, val updatedAt: Instant,
    val version: Long, val canWithdraw: Boolean, val counts: RunnerRemovalCountsDto,
    val timeline: List<RunnerTimelineStepDto> = emptyList(),
)
data class WithdrawRunnerRemovalCommand(@field:NotNull val expectedVersion: Long?)

data class StartRunnerLivenessCommand(
    @field:NotNull val eventId: UUID?,
    @field:AssertTrue @field:NotNull val biometricConsent: Boolean?,
    @field:NotBlank val consentVersion: String?,
)
