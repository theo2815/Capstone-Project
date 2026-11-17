package com.quickpitik.service.events

import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.RaceType
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals

// AA-R14: event lists say when a race's photographers charge different prices.
class EventDtoMapperTest {
    private val coverage = Mockito.mock(EventPhotographerRepository::class.java)
    private val mapper = EventDtoMapper(
        Mockito.mock(StorageService::class.java),
        StorageProperties(),
        Mockito.mock(PhotoRepository::class.java),
        coverage,
    )

    @Test
    fun `priceVaries only when a photographer with photos raised`() {
        val raised = event(RaceType.ROAD_RACE)
        val flat = event(RaceType.FUN_RUN)
        val personal = event(null)
        Mockito.`when`(coverage.findEventIdsWithRaisedPrice(anyArg())).thenReturn(listOf(raised.id))

        val dtos = mapper.toListDtos(listOf(raised, flat, personal)).associateBy { it.id }

        assertEquals(true, dtos.getValue(raised.id).priceVaries)
        assertEquals(false, dtos.getValue(flat.id).priceVaries)
        assertEquals(false, dtos.getValue(personal.id).priceVaries)
        assertEquals("ROAD_RACE", dtos.getValue(raised.id).eventType)
        assertEquals(null, dtos.getValue(personal.id).eventType)
        assertEquals(BigDecimal("125.00"), dtos.getValue(raised.id).pricePerPhoto)
        Mockito.verify(coverage, Mockito.times(1)).findEventIdsWithRaisedPrice(anyArg())

        val detail = mapper.toDetailDto(raised)
        assertEquals(true, detail.priceVaries)
        assertEquals("ROAD_RACE", detail.eventType)
    }

    private fun event(type: RaceType?) = Event(
        slug = "race-${UUID.randomUUID()}",
        name = "Cebu Run",
        date = LocalDate.of(2026, 12, 6),
        location = "Cebu City, Cebu",
        status = EventStatus.ACTIVE,
        pricePerPhoto = BigDecimal("125.00"),
    ).apply { eventType = type }

    private fun <T> anyArg(): T = Mockito.any()
}
