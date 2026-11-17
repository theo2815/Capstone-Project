package com.quickpitik.service.removal

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.AiProperties
import com.quickpitik.config.AiProvider
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.LivenessSessionRepository
import com.quickpitik.repository.PhotoRemovalRequestRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.events.EventFaceExpiry
import com.quickpitik.service.liveness.LivenessSessionService
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RunnerRemovalEligibilityServiceTest {
    private val events = mock(EventRepository::class.java)
    private val users = mock(UserRepository::class.java)
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val sessions = mock(LivenessSessionRepository::class.java)
    private val expiry = mock(EventFaceExpiry::class.java)
    private val liveness = mock(LivenessSessionService::class.java)
    private val userId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()
    private val clock = Clock.fixed(Instant.parse("2026-11-10T04:00:00Z"), ZoneOffset.UTC)

    // Manila midnight of the fixed clock's day.
    private val dayStart = java.time.ZonedDateTime.of(2026, 11, 10, 0, 0, 0, 0, java.time.ZoneId.of("Asia/Manila")).toOffsetDateTime()

    private fun service(
        aiEnabled: Boolean = true,
        provider: AiProvider = AiProvider.REKOGNITION,
        checksPerDay: Int = 3,
        maxOpen: Int = 3,
    ) = RunnerRemovalEligibilityService(
        events, users, requests, sessions, expiry,
        AiApiProperties(enabled = aiEnabled), AiProperties(provider),
        PhotoRemovalProperties(runner = PhotoRemovalProperties.Runner(checksPerDay = checksPerDay, maxOpenRequests = maxOpen)),
        clock,
    )

    private fun event(status: EventStatus = EventStatus.ACTIVE, deleted: Boolean = false) = Event(
        id = eventId, slug = "e-$eventId", name = "Fun Run", date = LocalDate.of(2026, 11, 1),
        location = "Cebu", status = status,
        deletedAt = if (deleted) OffsetDateTime.now() else null,
    )

    // Everything passes unless a test breaks one rule.
    private fun happy(event: Event = event(), verified: Boolean = true) {
        `when`(events.findById(eventId)).thenReturn(Optional.of(event))
        `when`(expiry.ended(event)).thenReturn(false)
        `when`(users.findById(userId)).thenReturn(Optional.of(
            User(email = "r@x.test", passwordHash = "x", name = "R", role = Role.RUNNER,
                emailVerifiedAt = if (verified) OffsetDateTime.now() else null)))
        `when`(requests.existsOpenRunnerRequest(userId, eventId)).thenReturn(false)
        `when`(requests.countOpenRunnerRequests(userId)).thenReturn(0L)
        `when`(sessions.countByUserIdAndCreatedAtGreaterThanEqual(userId, dayStart)).thenReturn(0L)
    }

    private fun assertNoAws() = verifyNoInteractions(liveness)

    @Test
    fun `an unknown event id is EVENT_NOT_ELIGIBLE, not eligible`() {
        `when`(events.findById(eventId)).thenReturn(Optional.empty())
        val r = service().check(userId, eventId)
        assertFalse(r.eligible)
        assertEquals(RunnerRemovalBlock.EVENT_NOT_ELIGIBLE, r.blockedReason)
        assertNoAws()
    }

    @Test
    fun `a draft or deleted event is EVENT_NOT_ELIGIBLE`() {
        for (e in listOf(event(EventStatus.DRAFT), event(deleted = true))) {
            `when`(events.findById(eventId)).thenReturn(Optional.of(e))
            assertEquals(RunnerRemovalBlock.EVENT_NOT_ELIGIBLE, service().check(userId, eventId).blockedReason)
        }
        assertNoAws()
    }

    @Test
    fun `AI off or a provider other than rekognition is AI_OFF`() {
        happy()
        assertEquals(RunnerRemovalBlock.AI_OFF, service(aiEnabled = false).check(userId, eventId).blockedReason)
        assertEquals(RunnerRemovalBlock.AI_OFF, service(provider = AiProvider.AI_API).check(userId, eventId).blockedReason)
        assertNoAws()
    }

    @Test
    fun `an event past its face retention is FACE_DATA_EXPIRED`() {
        val e = event(EventStatus.ARCHIVED)
        happy(e)
        `when`(expiry.ended(e)).thenReturn(true)
        assertEquals(RunnerRemovalBlock.FACE_DATA_EXPIRED, service().check(userId, eventId).blockedReason)
        assertNoAws()
    }

    @Test
    fun `checks-per-day 0 is INTAKE_CLOSED`() {
        happy()
        assertEquals(RunnerRemovalBlock.INTAKE_CLOSED, service(checksPerDay = 0).check(userId, eventId).blockedReason)
        assertNoAws()
    }

    @Test
    fun `REQUEST_OPEN carries the open request id`() {
        happy()
        val open = UUID.randomUUID()
        `when`(requests.existsOpenRunnerRequest(userId, eventId)).thenReturn(true)
        `when`(requests.findOpenRunnerRequestId(userId, eventId)).thenReturn(open)
        val r = service().check(userId, eventId)
        assertEquals(RunnerRemovalBlock.REQUEST_OPEN, r.blockedReason)
        assertEquals(open, r.openRequestId)
        assertNoAws()
    }

    @Test
    fun `reaching max-open-requests is TOO_MANY_OPEN`() {
        happy()
        `when`(requests.countOpenRunnerRequests(userId)).thenReturn(3L)
        assertEquals(RunnerRemovalBlock.TOO_MANY_OPEN, service(maxOpen = 3).check(userId, eventId).blockedReason)
        assertNoAws()
    }

    @Test
    fun `a used-up day is ATTEMPTS_EXHAUSTED with none left`() {
        happy()
        `when`(sessions.countByUserIdAndCreatedAtGreaterThanEqual(userId, dayStart)).thenReturn(3L)
        val r = service().check(userId, eventId)
        assertEquals(RunnerRemovalBlock.ATTEMPTS_EXHAUSTED, r.blockedReason)
        assertEquals(0, r.attemptsLeftToday)
        assertNoAws()
    }

    @Test
    fun `an unlisted public event is eligible`() {
        // Unlisted is a visibility, not a status: it passes (N3).
        happy(event().also { it.visibility = com.quickpitik.entity.EventVisibility.UNLISTED })
        val r = service().check(userId, eventId)
        assertTrue(r.eligible)
        assertNull(r.blockedReason)
        assertEquals(3, r.attemptsLeftToday)
        assertTrue(r.emailVerified)
    }

    @Test
    fun `an archived event inside face retention is eligible, and an unverified email is reported`() {
        happy(event(EventStatus.ARCHIVED), verified = false)
        val r = service().check(userId, eventId)
        assertTrue(r.eligible)
        assertFalse(r.emailVerified)
    }
}
