package com.quickpitik.service.cart

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.SavedEvent
import com.quickpitik.entity.SavedEventId
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.SavedEventRepository
import com.quickpitik.service.events.EventDtoMapper
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Runner-flow audit (2026-05-27), "Cart + saved events" items 2/3/4.
//
// save() used to accept any event id that existed, so a runner could save a
// DRAFT (admin-only pipeline state) or a soft-deleted event. Both write a row
// that list() then filters out — an invisible orphan the runner can neither see
// nor unsave. merge() had the same hole on the bulk path, but skips rather than
// throws (see the CartService.merge docblock for why).
class SavedEventsServiceTest {

    private lateinit var savedEventRepository: SavedEventRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var eventDtoMapper: EventDtoMapper
    private lateinit var service: SavedEventsService

    private val userId = UUID.randomUUID()
    private val saved = mutableListOf<SavedEvent>()

    @BeforeEach
    fun setUp() {
        savedEventRepository = Mockito.mock(SavedEventRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        eventDtoMapper = Mockito.mock(EventDtoMapper::class.java)
        service = SavedEventsService(savedEventRepository, eventRepository, eventDtoMapper)

        saved.clear()
        Mockito.`when`(savedEventRepository.save(anyArg<SavedEvent>())).thenAnswer {
            val row = it.arguments[0] as SavedEvent
            saved += row
            row
        }
        // list() re-reads after every mutation; an empty read keeps these tests
        // focused on what was written rather than on hydration.
        Mockito.`when`(savedEventRepository.findByUserIdOrderedBySavedAtDesc(userId))
            .thenReturn(emptyList())
    }

    @Test
    fun `save rejects a DRAFT event with 404`() {
        val event = event(status = EventStatus.DRAFT)
        Mockito.`when`(eventRepository.findById(event.id)).thenReturn(Optional.of(event))

        val ex = assertFailsWith<NotFoundException> { service.save(userId, event.id) }

        assertEquals(ErrorCodes.EVENT_NOT_FOUND, ex.code)
        assertEquals(emptyList(), saved)
    }

    @Test
    fun `save rejects a soft-deleted event with 404`() {
        val event = event(status = EventStatus.ACTIVE, deletedAt = OffsetDateTime.now())
        Mockito.`when`(eventRepository.findById(event.id)).thenReturn(Optional.of(event))

        val ex = assertFailsWith<NotFoundException> { service.save(userId, event.id) }

        assertEquals(ErrorCodes.EVENT_NOT_FOUND, ex.code)
        assertEquals(emptyList(), saved)
    }

    @Test
    fun `save accepts a live event`() {
        val event = event(status = EventStatus.ACTIVE)
        Mockito.`when`(eventRepository.findById(event.id)).thenReturn(Optional.of(event))
        Mockito.`when`(savedEventRepository.findById(SavedEventId(userId, event.id)))
            .thenReturn(Optional.empty())

        val dto = service.save(userId, event.id)

        assertEquals(event.id, dto.id)
        assertEquals(listOf(event.id), saved.map { it.id.eventId })
    }

    @Test
    fun `merge skips DRAFT and soft-deleted events but keeps the rest`() {
        val live = event(status = EventStatus.ACTIVE)
        val draft = event(status = EventStatus.DRAFT)
        val deleted = event(status = EventStatus.ACTIVE, deletedAt = OffsetDateTime.now())

        Mockito.`when`(savedEventRepository.findEventIdsByUserId(userId)).thenReturn(emptyList())
        Mockito.`when`(eventRepository.findAllById(anyArg<Iterable<UUID>>()))
            .thenReturn(listOf(live, draft, deleted))

        service.merge(userId, listOf(live.id, draft.id, deleted.id))

        assertEquals(listOf(live.id), saved.map { it.id.eventId })
    }

    @Test
    fun `merge does not re-save an event already saved`() {
        val live = event(status = EventStatus.ACTIVE)
        Mockito.`when`(savedEventRepository.findEventIdsByUserId(userId)).thenReturn(listOf(live.id))

        service.merge(userId, listOf(live.id))

        assertEquals(emptyList(), saved)
    }

    private fun event(
        status: EventStatus,
        deletedAt: OffsetDateTime? = null,
    ): Event = Event(
        slug = "cebu-marathon-${UUID.randomUUID()}",
        name = "Cebu Marathon",
        date = LocalDate.of(2026, 1, 11),
        location = "Cebu City, Cebu",
        status = status,
        deletedAt = deletedAt,
    )

    private fun <T> anyArg(): T = Mockito.any()
}
