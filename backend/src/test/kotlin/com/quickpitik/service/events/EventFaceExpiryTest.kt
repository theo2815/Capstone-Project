package com.quickpitik.service.events

import com.quickpitik.config.AiApiProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.repository.EventRepository
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// Legal review M2 (2026-11-04): an event's face templates are deleted 6 months
// after the event date (or when the event is deleted), and face search ends.
class EventFaceExpiryTest {

    private lateinit var eventRepository: EventRepository
    private lateinit var provider: FaceBibProvider
    private val events = mutableListOf<Event>()

    @BeforeEach
    fun setUp() {
        eventRepository = Mockito.mock(EventRepository::class.java)
        provider = Mockito.mock(FaceBibProvider::class.java)
        // Answer like the database would, from the in-memory rows.
        Mockito.`when`(eventRepository.findByFacesDeletedAtIsNullAndDateLessThanEqual(anyArg())).thenAnswer { inv ->
            val cutoff = inv.getArgument<LocalDate>(0)
            events.filter { it.facesDeletedAt == null && !it.date.isAfter(cutoff) }
        }
        Mockito.`when`(eventRepository.findByFacesDeletedAtIsNullAndDeletedAtIsNotNull()).thenAnswer {
            events.filter { it.facesDeletedAt == null && it.deletedAt != null }
        }
        Mockito.`when`(eventRepository.findById(anyArg())).thenAnswer { inv ->
            Optional.ofNullable(events.find { it.id == inv.getArgument<UUID>(0) })
        }
        Mockito.`when`(eventRepository.save(anyArg<Event>())).thenAnswer { it.arguments[0] }
    }

    private fun expiry(enabled: Boolean = true) =
        EventFaceExpiry(eventRepository, provider, AiApiProperties(enabled = enabled), CLOCK, testTransactionTemplate())

    private fun event(date: LocalDate, deleted: Boolean = false) = Event(
        slug = "run-${UUID.randomUUID().toString().take(6)}",
        name = "Run",
        date = date,
        location = "Cebu City",
        status = EventStatus.COMPLETED,
    ).also {
        if (deleted) it.deletedAt = OffsetDateTime.now(CLOCK)
        events += it
    }

    @Test
    fun `sweep deletes and stamps an event dated exactly 6 months ago`() {
        val old = event(TODAY.minusMonths(6))

        expiry().sweep()

        Mockito.verify(provider).deleteFacesByEvent(old.id)
        assertNotNull(old.facesDeletedAt)
    }

    @Test
    fun `sweep leaves a 5-month-old event alone`() {
        val recent = event(TODAY.minusMonths(5))

        expiry().sweep()

        Mockito.verify(provider, Mockito.never()).deleteFacesByEvent(recent.id)
        assertNull(recent.facesDeletedAt)
    }

    @Test
    fun `a failed delete leaves the stamp empty and the next run retries`() {
        val old = event(TODAY.minusMonths(7))
        Mockito.doThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "AWS down"))
            .doNothing()
            .`when`(provider).deleteFacesByEvent(old.id)

        expiry().sweep()
        assertNull(old.facesDeletedAt)

        expiry().sweep()
        assertNotNull(old.facesDeletedAt)
        Mockito.verify(provider, Mockito.times(2)).deleteFacesByEvent(old.id)
    }

    @Test
    fun `one failing event does not stop the others`() {
        val failing = event(TODAY.minusMonths(8))
        val fine = event(TODAY.minusMonths(9))
        Mockito.doThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "AWS down"))
            .`when`(provider).deleteFacesByEvent(failing.id)

        expiry().sweep()

        assertNull(failing.facesDeletedAt)
        assertNotNull(fine.facesDeletedAt)
    }

    @Test
    fun `sweep does nothing when AI is off`() {
        event(TODAY.minusYears(1))

        expiry(enabled = false).sweep()

        Mockito.verifyNoInteractions(provider)
        Mockito.verifyNoInteractions(eventRepository)
    }

    @Test
    fun `sweep deletes and stamps a deleted event however recent`() {
        val deleted = event(TODAY.minusDays(3), deleted = true)

        expiry().sweep()

        Mockito.verify(provider).deleteFacesByEvent(deleted.id)
        assertNotNull(deleted.facesDeletedAt)
    }

    @Test
    fun `face search ends at 6 months or once the templates are deleted`() {
        assertTrue(expiry().ended(event(TODAY.minusMonths(6))))
        assertFalse(expiry().ended(event(TODAY.minusMonths(6).plusDays(1))))
        assertTrue(expiry().ended(event(TODAY).apply { facesDeletedAt = OffsetDateTime.now(CLOCK) }))
        val byId = event(TODAY.minusYears(1))
        assertTrue(expiry().ended(byId.id))
        assertFalse(expiry().ended(UUID.randomUUID())) // unknown event: other checks answer 404
    }

    @Test
    fun `a deleted event counts as ended at once so no new enroll re-creates its collection`() {
        assertTrue(expiry().ended(event(TODAY, deleted = true)))
    }

    @Test
    fun `search ends on the same day the sweep deletes, even at month end`() {
        val endOfAugust = event(LocalDate.of(2026, 8, 31))
        val feb28 = Clock.fixed(Instant.parse("2027-02-28T04:00:00Z"), ZoneOffset.UTC)
        val mar1 = Clock.fixed(Instant.parse("2027-03-01T04:00:00Z"), ZoneOffset.UTC)
        fun at(clock: Clock) = EventFaceExpiry(eventRepository, provider, AiApiProperties(enabled = true), clock, testTransactionTemplate())

        assertFalse(at(feb28).ended(endOfAugust))
        at(feb28).sweep()
        assertNull(endOfAugust.facesDeletedAt)

        assertTrue(at(mar1).ended(endOfAugust))
        at(mar1).sweep()
        assertNotNull(endOfAugust.facesDeletedAt)
    }

    @Test
    fun `an old deleted event is deleted once`() {
        val both = event(TODAY.minusYears(1), deleted = true)

        expiry().sweep()

        Mockito.verify(provider, Mockito.times(1)).deleteFacesByEvent(both.id)
    }

    private fun <T> anyArg(): T = Mockito.any()

    private companion object {
        // 2026-11-03T20:00Z is already 2026-11-04 in Manila.
        val CLOCK: Clock = Clock.fixed(Instant.parse("2026-11-03T20:00:00Z"), ZoneOffset.UTC)
        val TODAY: LocalDate = LocalDate.of(2026, 11, 4)
    }
}
