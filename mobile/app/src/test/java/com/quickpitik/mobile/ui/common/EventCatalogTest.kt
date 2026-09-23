package com.quickpitik.mobile.ui.common

import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.LocalDate

class EventCatalogTest {
    @Test fun ownerWindowIncludesPreEventAndClampsToTheCalendarMonth() {
        val event = PhotographerEventSummaryDto("id", "slug", "Event", "2026-01-31", "Cebu", "open", 0, 0, 0.0,
            ownedByMe = true)
        for ((date, allowed) in listOf("2026-01-01" to true, "2026-02-28" to true, "2026-03-01" to false)) {
            assertEquals(date, allowed, canUploadToEvent(event, LocalDate.parse(date)))
        }
        for (review in listOf("pending", "rejected")) {
            assertEquals(false, canUploadToEvent(event.copy(reviewStatus = review), LocalDate.parse("2026-01-31")))
        }
        assertEquals(true, canUploadToEvent(event.copy(reviewStatus = "change_pending"), LocalDate.parse("2026-02-28")))
        assertEquals(false, canUploadToEvent(event.copy(ownedByMe = false), LocalDate.parse("2026-02-28")))
    }

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
