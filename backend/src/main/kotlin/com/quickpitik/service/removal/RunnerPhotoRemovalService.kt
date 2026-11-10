package com.quickpitik.service.removal

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.dto.photographer.LivenessSessionDto
import com.quickpitik.dto.removal.StartRunnerLivenessCommand
import com.quickpitik.entity.LivenessPurpose
import com.quickpitik.exception.ApiException
import com.quickpitik.service.liveness.LivenessSessionService
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import java.time.Clock
import java.time.Duration
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.UUID

/** Runner-side photo deletion requests. RD-2: starting the live check; RD-4 adds submit. */
@Service
class RunnerPhotoRemovalService(
    private val eligibilityService: RunnerRemovalEligibilityService,
    private val livenessSessionService: LivenessSessionService,
    private val legalProperties: LegalProperties,
    private val props: PhotoRemovalProperties,
    private val clock: Clock,
) {
    fun startLiveness(userId: UUID, command: StartRunnerLivenessCommand): LivenessSessionDto {
        val eventId = requireNotNull(command.eventId)
        eligibilityService.check(userId, eventId).blockedReason?.let { throw blockError(it) }
        if (command.biometricConsent != true) {
            throw ApiException(HttpStatus.BAD_REQUEST, ErrorCodes.VALIDATION_ERROR, "Please tick the box to continue.", "biometricConsent")
        }
        if (command.consentVersion != legalProperties.currentVersion) {
            throw ApiException(
                HttpStatus.CONFLICT, ErrorCodes.CONSENT_VERSION_MISMATCH,
                "The privacy notice was updated. Please reload the page and read it again.",
            )
        }
        return livenessSessionService.start(
            userId, LivenessPurpose.RUNNER_PHOTO_REMOVAL, eventId, props.runner.checksPerDay, command.consentVersion,
        )
    }

    // RD-4's submit re-check reuses this mapping.
    fun blockError(block: RunnerRemovalBlock): ApiException = when (block) {
        RunnerRemovalBlock.EVENT_NOT_ELIGIBLE ->
            ApiException(HttpStatus.NOT_FOUND, ErrorCodes.EVENT_NOT_FOUND, "We couldn't find that event.")
        RunnerRemovalBlock.AI_OFF ->
            ApiException(HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.AI_API_UNAVAILABLE, "Photo matching is not available right now.")
        RunnerRemovalBlock.FACE_DATA_EXPIRED ->
            ApiException(HttpStatus.GONE, ErrorCodes.FACE_SEARCH_ENDED, "Face search for this event has ended.")
        RunnerRemovalBlock.INTAKE_CLOSED ->
            ApiException(HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.REMOVAL_INTAKE_CLOSED, "Photo deletion requests are paused right now.")
        RunnerRemovalBlock.REQUEST_OPEN, RunnerRemovalBlock.TOO_MANY_OPEN ->
            ApiException(HttpStatus.CONFLICT, ErrorCodes.REMOVAL_REQUEST_OPEN, "You already have an open request.")
        RunnerRemovalBlock.ATTEMPTS_EXHAUSTED ->
            // Retry-After: seconds to the next Manila midnight, as AA's shared service answers it.
            ApiException(
                HttpStatus.TOO_MANY_REQUESTS, ErrorCodes.LIVENESS_ATTEMPTS_EXHAUSTED,
                "You have used today's live checks. Try again tomorrow.", retryAfterSeconds = secondsToManilaMidnight(),
            )
    }

    private fun secondsToManilaMidnight(): Long {
        val now = OffsetDateTime.now(clock.withZone(MANILA))
        return Duration.between(now, now.toLocalDate().plusDays(1).atStartOfDay(MANILA)).seconds.coerceAtLeast(1)
    }

    private companion object {
        val MANILA: ZoneId = ZoneId.of("Asia/Manila")
    }
}
