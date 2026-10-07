package com.quickpitik.service.photographer

import com.quickpitik.dto.photographer.PhotographerEventDetailDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
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
import kotlin.test.assertNotNull

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
    )
    private val seller = UUID.randomUUID()
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
