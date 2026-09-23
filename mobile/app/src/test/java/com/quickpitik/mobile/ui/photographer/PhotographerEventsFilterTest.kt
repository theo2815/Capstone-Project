package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import org.junit.Assert.assertEquals
import org.junit.Test

/** The roster's search box — the only thing that narrows the photographer's event list by text. */
class PhotographerEventsFilterTest {

    private fun event(name: String) =
        PhotographerEventSummaryDto("id-$name", "slug", name, "2026-09-12", "Cebu", "open", 0, 0, 0.0)

    private val roster = listOf(event("Cebu Marathon"), event("SINULOG RUN"), event("Talisay Trail"))

    @Test fun `a blank box shows everything, including whitespace-only`() {
        assertEquals(roster, filterEventsByName(roster, ""))
        assertEquals(roster, filterEventsByName(roster, "   "))
        assertEquals(emptyList<PhotographerEventSummaryDto>(), filterEventsByName(emptyList(), "cebu"))
    }

    @Test fun `the query is trimmed, case-insensitive, and matches anywhere in the name`() {
        assertEquals(listOf("Cebu Marathon"), filterEventsByName(roster, "  cebu ").map { it.name })
        assertEquals(listOf("SINULOG RUN"), filterEventsByName(roster, "sinulog").map { it.name })
        assertEquals(listOf("Talisay Trail"), filterEventsByName(roster, "TRAIL").map { it.name })
        // Mid-word, not just a prefix — finding one race in a long roster is the point.
        assertEquals(listOf("Cebu Marathon"), filterEventsByName(roster, "arath").map { it.name })
        // The location is NOT searched here (the runner's discovery screen does that).
        assertEquals(emptyList<String>(), filterEventsByName(roster, "Talisay City").map { it.name })
    }
}
