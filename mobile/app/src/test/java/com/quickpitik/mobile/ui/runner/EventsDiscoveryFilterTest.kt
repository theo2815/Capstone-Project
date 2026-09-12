package com.quickpitik.mobile.ui.runner

import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.ui.common.EventState
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Discovery's three filters over one list: city, lifecycle, free text. The
 * runner's only way to find a race in the public catalogue.
 */
class EventsDiscoveryFilterTest {

    private fun event(name: String, location: String, date: String, state: EventState) =
        EventDto("id-$name", "slug", name, date, location, null, 0, 0, "") to state

    private val catalogue = listOf(
        event("Cebu City Marathon", "Plaza Independencia, Cebu City", "2026-09-12", EventState.LIVE),
        event("Sinulog Run", "Fuente Osmena, Cebu City", "2026-11-01", EventState.UPCOMING),
        event("Talisay Trail", "Lantaw, Talisay", "2026-04-02", EventState.OPEN),
    )

    private fun names(events: List<Pair<EventDto, EventState>>) = events.map { it.first.name }

    @Test fun `no filter narrows nothing`() {
        assertEquals(catalogue, filterEvents(catalogue, "all", DateFilter.ANY, ""))
        assertEquals(catalogue, filterEvents(catalogue, "all", DateFilter.ANY, "   "))
    }

    @Test fun `city matches the tail of "Venue, City", not the venue`() {
        assertEquals(
            listOf("Cebu City Marathon", "Sinulog Run"),
            names(filterEvents(catalogue, "Cebu City", DateFilter.ANY, "")),
        )
        assertEquals(listOf("Talisay Trail"), names(filterEvents(catalogue, "Talisay", DateFilter.ANY, "")))
        // "Lantaw" is the venue half — it is never a city.
        assertEquals(emptyList<String>(), names(filterEvents(catalogue, "Lantaw", DateFilter.ANY, "")))
    }

    @Test fun `lifecycle filters on the display state, with Finished covering recent and archived`() {
        assertEquals(listOf("Cebu City Marathon"), names(filterEvents(catalogue, "all", DateFilter.LIVE, "")))
        assertEquals(listOf("Sinulog Run"), names(filterEvents(catalogue, "all", DateFilter.UPCOMING, "")))
        assertEquals(listOf("Talisay Trail"), names(filterEvents(catalogue, "all", DateFilter.FINISHED, "")))
        // Finished is both buckets — recent (OPEN) above and archived (PAST) here.
        val archived = listOf(event("Old Race", "Cebu City", "2020-01-01", EventState.PAST))
        assertEquals(listOf("Old Race"), names(filterEvents(archived, "all", DateFilter.FINISHED, "")))
        assertEquals(emptyList<String>(), names(filterEvents(archived, "all", DateFilter.LIVE, "")))
    }

    @Test fun `the query is case-insensitive and searches name, location and date together`() {
        assertEquals(listOf("Sinulog Run"), names(filterEvents(catalogue, "all", DateFilter.ANY, " SINULOG ")))
        // Location and date are part of the haystack — the photographer roster's
        // name-only filter is deliberately narrower.
        assertEquals(listOf("Talisay Trail"), names(filterEvents(catalogue, "all", DateFilter.ANY, "lantaw")))
        assertEquals(listOf("Cebu City Marathon"), names(filterEvents(catalogue, "all", DateFilter.ANY, "2026-09")))
        assertEquals(emptyList<String>(), names(filterEvents(catalogue, "all", DateFilter.ANY, "boracay")))
    }

    @Test fun `the three filters stack`() {
        assertEquals(
            listOf("Sinulog Run"),
            names(filterEvents(catalogue, "Cebu City", DateFilter.UPCOMING, "run")),
        )
        // Right city, right query, wrong lifecycle.
        assertEquals(emptyList<String>(), names(filterEvents(catalogue, "Cebu City", DateFilter.LIVE, "sinulog")))
    }
}
