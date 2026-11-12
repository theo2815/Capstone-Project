package com.quickpitik.service.removal

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.dto.photographer.LivenessSessionDto
import com.quickpitik.dto.removal.StartRunnerLivenessCommand
import com.quickpitik.entity.LivenessPurpose
import com.quickpitik.exception.ApiException
import com.quickpitik.service.liveness.LivenessSessionService
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import org.springframework.http.HttpStatus
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class RunnerPhotoRemovalLivenessTest {
    private val eligibility = mock(RunnerRemovalEligibilityService::class.java)
    private val liveness = mock(LivenessSessionService::class.java)
    private val userId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()
    private val ok = RunnerRemovalEligibility(true, null, 3, true, null)
    private val session = LivenessSessionDto("sess", "ap-southeast-5", 2, null)

    private fun service(checksPerDay: Int = 3) = RunnerPhotoRemovalService(
        eligibility, liveness, LegalProperties("1.3"),
        PhotoRemovalProperties(runner = PhotoRemovalProperties.Runner(checksPerDay = checksPerDay)),
        Clock.fixed(Instant.parse("2026-11-10T04:00:00Z"), ZoneOffset.UTC),
        // RD-4's submit collaborators; startLiveness never touches them.
        mock(com.quickpitik.repository.LivenessSessionRepository::class.java),
        @Suppress("UNCHECKED_CAST") (mock(org.springframework.beans.factory.ObjectProvider::class.java)
            as org.springframework.beans.factory.ObjectProvider<com.quickpitik.service.ai.RekognitionIdentityClient>),
        mock(RunnerAutoHidePolicy::class.java), mock(PhotoRemovalService::class.java), mock(PhotoRemovalEvidenceService::class.java),
        mock(com.quickpitik.service.photos.PhotoSearchService::class.java), mock(com.quickpitik.repository.PhotoRemovalRequestRepository::class.java),
        mock(com.quickpitik.repository.PhotoRemovalItemRepository::class.java), mock(com.quickpitik.repository.PhotoRepository::class.java),
        mock(com.quickpitik.repository.UserRepository::class.java), mock(com.quickpitik.repository.EventRepository::class.java),
        org.springframework.transaction.support.TransactionTemplate(),
        mock(com.quickpitik.service.admin.SystemFlagService::class.java),
    )

    private fun cmd(consent: Boolean = true, version: String = "1.3") = StartRunnerLivenessCommand(eventId, consent, version)

    private fun blocked(b: RunnerRemovalBlock, open: UUID? = null) =
        RunnerRemovalEligibility(false, b, 0, true, open)

    @Test
    fun `startLiveness passes RUNNER_PHOTO_REMOVAL, the event id, checks-per-day and the consent version to the shared service`() {
        `when`(eligibility.check(userId, eventId)).thenReturn(ok)
        `when`(liveness.start(userId, LivenessPurpose.RUNNER_PHOTO_REMOVAL, eventId, 3, "1.3")).thenReturn(session)
        assertEquals(session, service().startLiveness(userId, cmd()))
        // The version saved on the session row is the one the runner ticked.
        verify(liveness).start(userId, LivenessPurpose.RUNNER_PHOTO_REMOVAL, eventId, 3, "1.3")
    }

    @Test
    fun `an old consent version is refused with CONSENT_VERSION_MISMATCH before any AWS call`() {
        `when`(eligibility.check(userId, eventId)).thenReturn(ok)
        val ex = assertFailsWith<ApiException> { service().startLiveness(userId, cmd(version = "1.2")) }
        assertEquals(ErrorCodes.CONSENT_VERSION_MISMATCH, ex.code)
        assertEquals(HttpStatus.CONFLICT, ex.status)
        verifyNoInteractions(liveness)
    }

    @Test
    fun `no consent tick is refused before any AWS call`() {
        `when`(eligibility.check(userId, eventId)).thenReturn(ok)
        assertFailsWith<ApiException> { service().startLiveness(userId, cmd(consent = false)) }
        verifyNoInteractions(liveness)
    }

    @Test
    fun `checks-per-day 0 answers REMOVAL_INTAKE_CLOSED`() {
        `when`(eligibility.check(userId, eventId)).thenReturn(blocked(RunnerRemovalBlock.INTAKE_CLOSED))
        val ex = assertFailsWith<ApiException> { service(checksPerDay = 0).startLiveness(userId, cmd()) }
        assertEquals(ErrorCodes.REMOVAL_INTAKE_CLOSED, ex.code)
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, ex.status)
        verifyNoInteractions(liveness)
    }

    @Test
    fun `every block maps to its error and none reaches AWS`() {
        val expected = mapOf(
            RunnerRemovalBlock.EVENT_NOT_ELIGIBLE to (HttpStatus.NOT_FOUND to ErrorCodes.EVENT_NOT_FOUND),
            RunnerRemovalBlock.AI_OFF to (HttpStatus.SERVICE_UNAVAILABLE to ErrorCodes.AI_API_UNAVAILABLE),
            RunnerRemovalBlock.FACE_DATA_EXPIRED to (HttpStatus.GONE to ErrorCodes.FACE_SEARCH_ENDED),
            RunnerRemovalBlock.INTAKE_CLOSED to (HttpStatus.SERVICE_UNAVAILABLE to ErrorCodes.REMOVAL_INTAKE_CLOSED),
            RunnerRemovalBlock.REQUEST_OPEN to (HttpStatus.CONFLICT to ErrorCodes.REMOVAL_REQUEST_OPEN),
            RunnerRemovalBlock.TOO_MANY_OPEN to (HttpStatus.CONFLICT to ErrorCodes.REMOVAL_REQUEST_OPEN),
            RunnerRemovalBlock.ATTEMPTS_EXHAUSTED to (HttpStatus.TOO_MANY_REQUESTS to ErrorCodes.LIVENESS_ATTEMPTS_EXHAUSTED),
        )
        assertEquals(RunnerRemovalBlock.entries.toSet(), expected.keys)
        for ((block, want) in expected) {
            `when`(eligibility.check(userId, eventId)).thenReturn(blocked(block))
            val ex = assertFailsWith<ApiException> { service().startLiveness(userId, cmd()) }
            assertEquals(want, ex.status to ex.code, "block $block")
            if (block == RunnerRemovalBlock.ATTEMPTS_EXHAUSTED) assertEquals(true, (ex.retryAfterSeconds ?: 0) > 0)
        }
        verifyNoInteractions(liveness)
    }

    @Test
    fun `a throttled AWS create answers LIVENESS_UNAVAILABLE`() {
        `when`(eligibility.check(userId, eventId)).thenReturn(ok)
        `when`(liveness.start(userId, LivenessPurpose.RUNNER_PHOTO_REMOVAL, eventId, 3, "1.3"))
            .thenThrow(ApiException(HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.LIVENESS_UNAVAILABLE, "busy"))
        val ex = assertFailsWith<ApiException> { service().startLiveness(userId, cmd()) }
        assertEquals(ErrorCodes.LIVENESS_UNAVAILABLE, ex.code)
    }
}
