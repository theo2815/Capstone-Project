package com.quickpitik.mobile.ui.runner

import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.LocalDate

class EventCatalogTest {
    @Test
    fun raceDayDisplayDoesNotCloseTheUploadAndRealtimeWindow() {
        val race = LocalDate.of(2026, 9, 9)
        for (days in listOf(-1L, 0L, 1L, 3L, 4L, 90L)) {
            val today = race.plusDays(days)
            val lifecycle = when {
                days < 0 -> EventState.UPCOMING
                days <= 3 -> EventState.LIVE
                days < 90 -> EventState.OPEN
                else -> EventState.PAST
            }
            val display = if (days in 1L..3L) EventState.OPEN else lifecycle
            assertEquals("lifecycle day $days", lifecycle, deriveEventState(race.toString(), today))
            assertEquals("display day $days", display, deriveEventDisplayState(race.toString(), today))
            assertEquals("upload day $days", days in 0L..3L, canUploadToEvent(race.toString(), today))
        }
    }

    @Test
    fun locationFilterUsesTheLocationTailWithoutGuessingARegion() {
        for ((location, city) in listOf(
            "Carcar City, Cebu" to "Cebu",
            "Cagayan de Oro, Misamis Oriental" to "Misamis Oriental",
            "Butuan, Caraga" to "Caraga",
            "  An unlisted town  " to "An unlisted town",
        )) {
            assertEquals(city, extractCity(location))
        }
    }
}
