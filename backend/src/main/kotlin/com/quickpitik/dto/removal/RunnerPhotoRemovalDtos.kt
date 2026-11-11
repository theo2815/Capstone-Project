package com.quickpitik.dto.removal

import jakarta.validation.constraints.AssertTrue
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotNull
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

data class StartRunnerLivenessCommand(
    @field:NotNull val eventId: UUID?,
    @field:AssertTrue @field:NotNull val biometricConsent: Boolean?,
    @field:NotBlank val consentVersion: String?,
)
