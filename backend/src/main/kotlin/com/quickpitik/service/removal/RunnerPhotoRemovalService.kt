package com.quickpitik.service.removal

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.dto.photographer.LivenessSessionDto
import com.quickpitik.dto.removal.RunnerRemovalReason
import com.quickpitik.dto.removal.RunnerSubmitResultDto
import com.quickpitik.dto.removal.SearchImage
import com.quickpitik.dto.removal.StartRunnerLivenessCommand
import com.quickpitik.dto.removal.SubmitRunnerRemovalCommand
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.RekognitionIdentityClient
import com.quickpitik.service.liveness.LivenessSessionService
import com.quickpitik.service.photos.PhotoSearchService
import com.quickpitik.service.profile.SelfieService
import org.springframework.beans.factory.ObjectProvider
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
import software.amazon.awssdk.core.exception.SdkException
import java.math.BigDecimal
import java.math.RoundingMode
import java.time.Clock
import java.time.Duration
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.UUID

/** Runner-side photo deletion requests. RD-2: starting the live check; RD-4: submit. */
@Service
class RunnerPhotoRemovalService(
    private val eligibilityService: RunnerRemovalEligibilityService,
    private val livenessSessionService: LivenessSessionService,
    private val legalProperties: LegalProperties,
    private val props: PhotoRemovalProperties,
    private val clock: Clock,
    private val sessions: LivenessSessionRepository,
    private val identityClient: ObjectProvider<RekognitionIdentityClient>,
    private val policy: RunnerAutoHidePolicy,
    private val removals: PhotoRemovalService,
    private val evidence: PhotoRemovalEvidenceService,
    private val search: PhotoSearchService,
    private val requests: PhotoRemovalRequestRepository,
    private val items: PhotoRemovalItemRepository,
    private val photos: PhotoRepository,
    private val users: UserRepository,
    private val events: EventRepository,
    private val tx: TransactionTemplate,
) {
    /**
     * Turns a passed live check into a case. Every AWS and R2 call runs outside a transaction; the
     * evidence key is stored before the upload, so a crash leaves a key the cleanup sweep deletes.
     */
    @Transactional(propagation = Propagation.NEVER)
    fun submit(userId: UUID, key: UUID, command: SubmitRunnerRemovalCommand): RunnerSubmitResultDto {
        replay(userId, key)?.let { return it }
        val eventId = requireNotNull(command.eventId)
        val sessionId = requireNotNull(command.sessionId)
        val reason = RunnerRemovalReason.entries.firstOrNull { it.name == command.reasonCode }
            ?: throw ApiException(HttpStatus.BAD_REQUEST, ErrorCodes.VALIDATION_ERROR, "Please pick a reason.", "reasonCode")
        // The session already counted toward today, so a used-up day still lets this one through.
        eligibilityService.check(userId, eventId).blockedReason
            ?.takeIf { it != RunnerRemovalBlock.ATTEMPTS_EXHAUSTED }?.let { throw blockError(it) }

        if (!livenessSessionService.claim(userId, sessionId, LivenessPurpose.RUNNER_PHOTO_REMOVAL, eventId)) {
            replay(userId, key)?.let { return it } // a racing twin with the same key
            throw LivenessSessionService.invalidSession()
        }
        val session = checkNotNull(sessions.findBySessionIdAndUserIdAndPurpose(sessionId, userId, LivenessPurpose.RUNNER_PHOTO_REMOVAL))
        val client = identityClient.getIfAvailable() ?: throw livenessUnavailable()
        val result = try { client.livenessResult(sessionId) } catch (error: SdkException) { throw livenessUnavailable() }
        val confidence = result.confidence?.toDouble()
        val band = policy.livenessBand(confidence, result.status.toString())
            ?: throw ApiException(HttpStatus.UNPROCESSABLE_ENTITY, ErrorCodes.LIVENESS_FAILED, "The live check did not pass. Please try again.")
        // shortcut: assumes AWS returns a JPEG reference image within the selfie size cap; RD-0.3 records the real facts.
        val bytes = result.referenceImage
        if (bytes == null || bytes.size < 2 || bytes.size > SelfieService.MAX_SELFIE_BYTES ||
            bytes[0] != 0xFF.toByte() || bytes[1] != 0xD8.toByte()) throw livenessUnavailable()
        val image = SearchImage(bytes, "image/jpeg", "liveness-reference.jpg")

        val runner = users.findById(userId).orElseThrow()
        val event = events.findById(eventId).orElseThrow()
        val draft = removals.openRunnerDraft(RunnerDraftCommand(UUID.randomUUID(), userId, runner.email, event, key, reason.name,
            command.details?.takeIf { it.isNotBlank() }, checkNotNull(session.consentVersion), session.createdAt.toInstant(), sessionId,
            band, BigDecimal.valueOf(confidence!!).setScale(2, RoundingMode.HALF_UP), image.contentType))
        evidence.putSystemEvidence(draft.evidenceKey, image)
        val case = removals.completeRunnerCase(draft.requestId, match(eventId, image, runner.emailVerifiedAt != null,
            requests.existsRejectedRunnerRequest(userId, eventId)))
        return RunnerSubmitResultDto(draft.requestId, statusOf(draft.requestId), case.hiddenCount, case.suggestedCount)
    }

    private fun match(eventId: UUID, image: SearchImage, emailVerified: Boolean, priorRejection: Boolean): RunnerMatchOutcome {
        val found = try {
            search.findFaceCandidates(eventId, image, props.runner.candidateLimit)
        } catch (error: AiApiException) {
            return RunnerMatchOutcome(true, emptyList(), false, emailVerified, priorRejection)
        } catch (error: ApiException) {
            // Face data expired after the eligibility check.
            if (error.code != ErrorCodes.FACE_SEARCH_ENDED) throw error
            return RunnerMatchOutcome(true, emptyList(), false, emailVerified, priorRejection)
        }
        // Event isolation: only this event's "<eventId>.<photoId>" person ids reach the photo lookup.
        val scores = found.matches.filter { it.personId.startsWith("$eventId.") }
            .groupBy { it.personId }.mapValues { (_, m) -> m.maxOf { it.similarity } }
        val candidates = if (found.noFaces || scores.isEmpty()) emptyList() else tx.execute {
            photos.findRemovalReviewCandidates(eventId, scores.keys).filter { it.removalState != RemovalState.REMOVED }.mapNotNull { photo ->
                photo.facePersons.mapNotNull { scores[it.aiPersonId] }.maxOrNull()
                    ?.let { CandidatePhoto(photo.id, photo.photographerId, it, photo.faceCount?.toInt()) }
            }
        }!!
        return RunnerMatchOutcome(false, candidates, found.possiblyTruncated, emailVerified, priorRejection)
    }

    private fun replay(userId: UUID, key: UUID): RunnerSubmitResultDto? {
        val request = requests.findByCreationCommandId(key) ?: return null
        if (request.source != RemovalRequestSource.RUNNER || request.requesterUserId != userId) removalConflict()
        val all = items.findByRequestIdOrderByIdAsc(request.id)
        return RunnerSubmitResultDto(request.id, statusOf(request, all),
            all.count { it.state == RemovalItemState.VERIFIED }, all.count { it.state == RemovalItemState.CANDIDATE })
    }

    private fun statusOf(requestId: UUID): String =
        statusOf(requests.findById(requestId).orElseThrow(), items.findByRequestIdOrderByIdAsc(requestId))

    // shortcut: RD-5's status rules, inline until RD-5 replaces this with RunnerRemovalStatus.of.
    private fun statusOf(r: PhotoRemovalRequest, all: List<PhotoRemovalItem>): String {
        val undecided = all.any { it.state == RemovalItemState.CANDIDATE || (it.state == RemovalItemState.VERIFIED && it.adminDecision == null) }
        val settled = all.any { it.state == RemovalItemState.REMOVAL_AUTHORIZED || it.state == RemovalItemState.REMOVED }
        return when {
            r.closeReason == RunnerRemovalCloseReason.WITHDRAWN -> "WITHDRAWN"
            r.closeReason == RunnerRemovalCloseReason.NO_MATCHES -> "NO_MATCHES_FOUND"
            r.status == RemovalRequestStatus.DRAFT && all.isEmpty() && RunnerRiskFlag.MATCHING_UNAVAILABLE.name in r.riskFlags -> "PROCESSING"
            all.any { it.state == RemovalItemState.VERIFIED && it.adminDecision == RemovalDecision.MORE_EVIDENCE } ||
                (r.livenessBand == RunnerLivenessBand.UNCERTAIN && undecided && !settled) -> "ADDITIONAL_VERIFICATION_REQUIRED"
            undecided -> "PENDING_ADMIN_REVIEW"
            all.any { it.state == RemovalItemState.REMOVAL_AUTHORIZED } -> "APPROVED"
            all.any { it.state == RemovalItemState.REMOVED } -> "COMPLETED"
            else -> "REJECTED"
        }
    }

    private fun livenessUnavailable() =
        ApiException(HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.LIVENESS_UNAVAILABLE, "The live check is not available right now.")

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
