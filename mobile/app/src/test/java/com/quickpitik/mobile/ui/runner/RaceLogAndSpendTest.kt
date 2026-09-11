package com.quickpitik.mobile.ui.runner

import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.OrderListItemDto
import com.quickpitik.mobile.data.remote.SavedEventSummaryDto
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The runner's history views: the profile race log (saved ∪ purchased, one row
 * per event) and the orders page's lifetime spend. Display-only, but both
 * add up money, so the sums and the de-duplication are pinned.
 */
class RaceLogAndSpendTest {
    @Test fun `race log merges saves and orders per event, sums purchases, newest race first`() {
        val log = buildRaceLog(
            saved = listOf(
                saved("e1", "Saved Name", "2026-08-01", banner = "s1.jpg"),
                saved("e2", "Night Run", "2026-10-01"),
            ),
            orders = listOf(
                order("e1", listOf("a", "b"), 100.0, name = "Cebu Run", date = "2026-08-01"),
                order("e1", listOf("c"), 50.0, name = "Cebu Run", date = "2026-08-01"),
                order("e3", listOf("d"), 25.0), // no name or date on the order
                order("e4", listOf("e"), 10.0),
            ),
            eventCatalog = mapOf("e3" to event("e3", "Mactan Run", "2026-07-01", "e3.jpg")),
        )

        assertEquals(listOf("e2", "e1", "e3", "e4"), log.map { it.eventId })
        val (nightRun, cebu, mactan, unknown) = log
        assertEquals(RaceLogEntry("e1", "Cebu Run", "e1", "2026-08-01", "s1.jpg", "Cebu", 3, 150.0, saved = true, purchased = true), cebu)
        assertEquals(RaceLogEntry("e2", "Night Run", "e2", "2026-10-01", null, "Cebu", 0, 0.0, saved = true, purchased = false), nightRun)
        assertEquals(RaceLogEntry("e3", "Mactan Run", "e3", "2026-07-01", "e3.jpg", "Cebu", 1, 25.0, saved = false, purchased = true), mactan)
        assertEquals("Marathon Event", unknown.eventName)
        assertEquals(null, unknown.eventDate)
    }

    @Test fun `lifetime spend sums every order and dates from the earliest readable payment`() {
        assertEquals(SpendStats(0.0, 0, 0, null), computeSpendStats(emptyList()))

        val stats = computeSpendStats(
            listOf(
                order("e1", listOf("a", "b"), 100.0, paidAt = "2026-03-05T10:00:00+08:00"),
                order("e2", listOf("c"), 49.5, paidAt = "2025-11-20T09:00:00Z"),
                order("e3", listOf("d"), 20.0, paidAt = "not a date"),
                order("e4", listOf("e", "f"), 0.0, paidAt = null),
            ),
        )
        assertEquals(SpendStats(169.5, 4, 6, "Nov 2025"), stats)

        // "Since" is Manila's month: 20:00 UTC on Nov 30 is already Dec 1 there.
        val boundary = computeSpendStats(listOf(order("e5", listOf("g"), 1.0, paidAt = "2025-11-30T20:00:00Z")))
        assertEquals("Dec 2025", boundary.firstPurchase)
    }

    private fun order(
        eventId: String,
        photoIds: List<String>,
        total: Double,
        name: String? = null,
        date: String? = null,
        paidAt: String? = null,
    ) = OrderListItemDto(
        id = "o-$eventId-${photoIds.first()}", eventId = eventId, photoIds = photoIds, total = total,
        paymentMethod = "gcash", paidAt = paidAt, eventName = name, eventSlug = name?.let { eventId },
        eventDate = date, eventState = null, status = "PAID",
    )

    private fun saved(id: String, name: String, date: String, banner: String? = null) =
        SavedEventSummaryDto(id, id, name, date, "ACTIVE", banner, "Cebu", "2026-01-01T00:00:00Z")

    private fun event(id: String, name: String, date: String, banner: String) =
        EventDto(id, id, name, date, "Cebu", banner, 0, 0, "COMPLETED")
}
