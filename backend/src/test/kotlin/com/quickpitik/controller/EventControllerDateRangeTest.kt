package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.dto.events.EventDto
import com.quickpitik.exception.ValidationException
import com.quickpitik.service.events.EventService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Runner-flow audit (2026-05-27), "Events + photo discovery" item E3.
//
// EventRepository.search filters `date >= :dateFrom AND date <= :dateTo`, so a
// reversed range matched nothing and the runner read the empty page as "no
// events here" instead of "your filter is backwards".
class EventControllerDateRangeTest {

    private lateinit var eventService: EventService
    private lateinit var controller: EventController

    @BeforeEach
    fun setUp() {
        eventService = Mockito.mock(EventService::class.java)
        Mockito.`when`(eventService.list(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(PaginatedResponse(emptyList<EventDto>(), 0L, 0, 20))
        controller = EventController(eventService)
    }

    @Test
    fun `dateFrom after dateTo is rejected against the dateTo field`() {
        val ex = assertFailsWith<ValidationException> {
            list(dateFrom = "2026-12-01", dateTo = "2026-01-01")
        }

        assertEquals(ErrorCodes.VALIDATION_ERROR, ex.code)
        assertEquals("dateTo", ex.field)
        Mockito.verify(eventService, Mockito.never())
            .list(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `an identical dateFrom and dateTo is a valid single-day filter`() {
        list(dateFrom = "2026-01-11", dateTo = "2026-01-11")

        Mockito.verify(eventService).list(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `an open-ended range on either side still passes through`() {
        list(dateFrom = "2026-12-01", dateTo = null)
        list(dateFrom = null, dateTo = "2026-01-01")
        list(dateFrom = null, dateTo = null)

        Mockito.verify(eventService, Mockito.times(3))
            .list(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `a malformed date is still reported against its own field`() {
        val ex = assertFailsWith<ValidationException> { list(dateFrom = "11-01-2026", dateTo = null) }

        assertEquals("dateFrom", ex.field)
    }

    private fun list(dateFrom: String?, dateTo: String?) = controller.list(
        status = null,
        search = null,
        city = null,
        dateFrom = dateFrom,
        dateTo = dateTo,
        offset = null,
        limit = null,
    )

    private fun <T> anyArg(): T = Mockito.any()
}
