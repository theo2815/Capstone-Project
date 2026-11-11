package com.quickpitik.service.photographer

import com.quickpitik.dto.photographer.PhotographerEventDetailDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.RaceType
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.config.StorageProperties
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class PhotographerEventServiceTest {
    private val events = Mockito.mock(EventRepository::class.java)
    private val coverage = Mockito.mock(EventPhotographerRepository::class.java)
    private val purchases = Mockito.mock(OrderItemRepository::class.java)
    private val service = PhotographerEventService(
        coverage,
        events,
        Mockito.mock(PhotoRepository::class.java),
        purchases,
        PhotographerPhotoDtoMapper(
            Mockito.mock(StorageService::class.java),
            Mockito.mock(StorageProperties::class.java),
        ),
        Mockito.mock(StorageService::class.java),
        Mockito.mock(StorageProperties::class.java),
        Mockito.mock(EventDtoMapper::class.java),
        Mockito.mock(PhotographerSettingsRepository::class.java),
        com.quickpitik.service.events.RacePriceService(
            events, coverage, Mockito.mock(PhotoRepository::class.java),
            Mockito.mock(com.quickpitik.repository.RacePriceChangeRepository::class.java),
            Mockito.mock(com.quickpitik.service.photographer.PhotographerAccessGuard::class.java),
            com.quickpitik.config.RaceProperties(selfServeEnabled = true), com.quickpitik.config.PlatformProperties(),
            java.time.Clock.systemUTC(),
            Mockito.mock(com.quickpitik.service.admin.AdminDecisionLogService::class.java),
        ),
        com.quickpitik.config.RaceProperties(selfServeEnabled = true),
    )
    private val seller = UUID.randomUUID()

    private fun covering(event: Event) = Mockito.`when`(coverage.findById(EventPhotographerId(event.id, seller)))
        .thenReturn(Optional.of(com.quickpitik.entity.EventPhotographer(id = EventPhotographerId(event.id, seller))))
    private val removedAt = OffsetDateTime.parse("2026-09-04T21:36:27Z")

    @Test
    fun `removed owned event requires explicit history opt in`() {
        val event = event(createdBy = seller, deletedAt = removedAt)
        Mockito.`when`(events.findById(event.id)).thenReturn(Optional.of(event))

        assertFailsWith<NotFoundException> { service.getEventDetail(seller, event.id) }
        assertEquals(removedAt, historyDetail(seller, event.id).deletedAt)
    }

    @Test
    fun `live and archived event details remain available without opt in`() {
        for (status in listOf(EventStatus.ACTIVE, EventStatus.ARCHIVED)) {
            val event = event(status = status, createdBy = seller)
            Mockito.`when`(events.findById(event.id)).thenReturn(Optional.of(event))
            assertEquals(event.id, service.getEventDetail(seller, event.id).id)
        }
    }

    // ─── AA-R3: a typed race opens its detail for a joiner or its submitter before any upload ───

    @Test
    fun `a joined race with no uploads opens its detail`() {
        val race = event(status = EventStatus.ACTIVE).apply { eventType = RaceType.ROAD_RACE }
        Mockito.`when`(events.findById(race.id)).thenReturn(Optional.of(race))
        covering(race)

        val detail = service.getEventDetail(seller, race.id)

        assertEquals(race.id, detail.id)
        assertEquals("ROAD_RACE", detail.eventType)
        assertFalse(detail.submittedByMe)
        assertFalse(detail.ownedByMe)
    }

    // AA-R11: the detail carries the photographer's race price card only on a race they cover.
    @Test
    fun `a covered race's detail carries the race price and an uncovered one does not`() {
        val race = event(status = EventStatus.ACTIVE).apply { eventType = RaceType.ROAD_RACE; submittedBy = seller }
        Mockito.`when`(events.findById(race.id)).thenReturn(Optional.of(race))
        assertNull(service.getEventDetail(seller, race.id).racePrice, "the submitter without a coverage row")

        covering(race)
        val price = assertNotNull(service.getEventDetail(seller, race.id).racePrice)
        assertEquals(0, race.pricePerPhoto.compareTo(price.floorPhp))
    }

    @Test
    fun `the submitter's race with no uploads opens its detail`() {
        val race = event(status = EventStatus.ACTIVE).apply { eventType = RaceType.TRAIL_RUN; submittedBy = seller }
        Mockito.`when`(events.findById(race.id)).thenReturn(Optional.of(race))

        val detail = service.getEventDetail(seller, race.id)

        assertTrue(detail.submittedByMe)
        assertFalse(detail.ownedByMe, "a race is shared: ownedByMe stays createdBy == viewer")
    }

    @Test
    fun `the submitter can edit their race until another photographer uploads`() {
        val race = event(status = EventStatus.ACTIVE).apply { eventType = RaceType.TRAIL_RUN; submittedBy = seller }
        Mockito.`when`(events.findById(race.id)).thenReturn(Optional.of(race))
        val joiner = UUID.randomUUID()
        Mockito.`when`(coverage.findById(EventPhotographerId(race.id, joiner)))
            .thenReturn(Optional.of(com.quickpitik.entity.EventPhotographer(id = EventPhotographerId(race.id, joiner))))

        assertTrue(service.getEventDetail(seller, race.id).canEditRace)
        assertFalse(service.getEventDetail(joiner, race.id).canEditRace, "only the submitter edits")

        Mockito.`when`(coverage.existsOtherUploader(race.id, seller)).thenReturn(true)
        assertFalse(service.getEventDetail(seller, race.id).canEditRace)
    }

    @Test
    fun `an untyped event with no coverage still answers not found`() {
        val adminEvent = event(status = EventStatus.ACTIVE)
        Mockito.`when`(events.findById(adminEvent.id)).thenReturn(Optional.of(adminEvent))
        covering(adminEvent)

        assertFailsWith<NotFoundException> { service.getEventDetail(seller, adminEvent.id) }
    }

    private fun historyDetail(photographerId: UUID, eventId: UUID): PhotographerEventDetailDto {
        val method = PhotographerEventService::class.java.methods.singleOrNull {
            it.name == "getEventDetail" && it.parameterTypes.contentEquals(arrayOf(UUID::class.java, UUID::class.java, Boolean::class.javaPrimitiveType))
        }
        assertNotNull(method, "private history needs an explicit includeRemoved argument")
        return try {
            method.invoke(service, photographerId, eventId, true) as PhotographerEventDetailDto
        } catch (error: java.lang.reflect.InvocationTargetException) {
            throw error.targetException
        }
    }

    private fun event(
        status: EventStatus = EventStatus.ARCHIVED,
        createdBy: UUID? = null,
        deletedAt: OffsetDateTime? = null,
    ) = Event(
        slug = "history-${UUID.randomUUID()}",
        name = "History",
        date = LocalDate.of(2026, 9, 1),
        location = "Cebu",
        status = status,
        createdBy = createdBy,
        deletedAt = deletedAt,
    )
}
