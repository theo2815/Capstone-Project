package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceProperties
import org.junit.jupiter.api.Test
import java.time.LocalDate
import java.time.LocalDateTime
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RelevanceRulesTest {

    private val copy = RelevanceProperties.Copy()
    private val date = RelevanceProperties.Date()
    private val eventDate = LocalDate.of(2026, 9, 27)

    private fun at(text: String) = LocalDateTime.parse(text)

    @Test fun `three hits in sixty look-ups is low and three in sixty one is not`() {
        assertTrue(copyLow(tightHits = 3, lookups = 60, p = copy))
        assertFalse(copyLow(tightHits = 3, lookups = 61, p = copy))
    }

    @Test fun `two hits are never low`() {
        assertFalse(copyLow(tightHits = 2, lookups = 2, p = copy))
    }

    @Test fun `no look-ups is not low`() {
        assertFalse(copyLow(tightHits = 0, lookups = 0, p = copy))
    }

    @Test fun `a capture time with no offset is read as Manila time`() {
        assertEquals(LocalDate.of(2026, 9, 27), manilaDate(at("2026-09-27T23:59:59"), null))
        assertEquals(LocalDate.of(2026, 9, 28), manilaDate(at("2026-09-28T00:00:00"), null))
        assertEquals(LocalDate.of(2026, 9, 27), manilaDate(at("2026-09-27T23:59:59"), ""))
    }

    @Test fun `an offset moves the date across midnight`() {
        assertEquals(LocalDate.of(2026, 9, 28), manilaDate(at("2026-09-27T20:30"), "Z"))
        assertEquals(LocalDate.of(2026, 9, 27), manilaDate(at("2026-09-28T00:30"), "+09:00"))
        assertEquals(LocalDate.of(2026, 9, 27), manilaDate(at("2026-09-27T15:59:59"), "Z"))
        assertEquals(LocalDate.of(2026, 9, 28), manilaDate(at("2026-09-27T16:00:00"), "Z"))
    }

    @Test fun `a Manila offset does not move the date`() {
        assertEquals(LocalDate.of(2026, 9, 27), manilaDate(at("2026-09-27T23:59:59"), "+08:00"))
        assertEquals(LocalDate.of(2026, 9, 28), manilaDate(at("2026-09-28T00:00:00"), "+08:00"))
    }

    @Test fun `an offset that does not parse falls back to the camera date`() {
        assertEquals(LocalDate.of(2026, 9, 27), manilaDate(at("2026-09-27T20:30"), "nonsense"))
        assertEquals(LocalDate.of(2026, 9, 27), manilaDate(at("2026-09-27T20:30"), "+25:00"))
    }

    @Test fun `two days either side of the event date is inside and three is outside`() {
        assertFalse(outsideWindow(eventDate.minusDays(2), eventDate, date.windowDays))
        assertFalse(outsideWindow(eventDate.plusDays(2), eventDate, date.windowDays))
        assertTrue(outsideWindow(eventDate.minusDays(3), eventDate, date.windowDays))
        assertTrue(outsideWindow(eventDate.plusDays(3), eventDate, date.windowDays))
    }

    @Test fun `the last minute of day two is inside and the first of day three is outside, through an offset`() {
        // 15:59 UTC is 23:59 in Manila; 16:00 UTC is midnight.
        assertFalse(outsideWindow(manilaDate(at("2026-09-29T15:59"), "Z"), eventDate, date.windowDays))
        assertTrue(outsideWindow(manilaDate(at("2026-09-29T16:00"), "Z"), eventDate, date.windowDays))
    }

    @Test fun `eight of ten dated photos outside is low and seven of ten is not`() {
        assertEquals(true, dateLow(dated = 10, outside = 8, pending = 0, p = date))
        assertEquals(false, dateLow(dated = 10, outside = 7, pending = 0, p = date))
    }

    @Test fun `fewer than ten dated photos is never low`() {
        assertEquals(false, dateLow(dated = 9, outside = 9, pending = 0, p = date))
    }

    @Test fun `pending metadata makes the date rule wait`() {
        assertNull(dateLow(dated = 10, outside = 10, pending = 1, p = date))
    }

    @Test fun `a shared race closes three days after its date and an own event one month after`() {
        assertEquals(LocalDate.of(2026, 9, 30), lastUploadDate(eventDate, sharedRace = true))
        assertEquals(LocalDate.of(2026, 10, 27), lastUploadDate(eventDate, sharedRace = false))
        assertEquals(LocalDate.of(2026, 2, 28), lastUploadDate(LocalDate.of(2026, 1, 31), sharedRace = false))
    }

    @Test fun `the admin cutoff is the last date whose uploads have closed`() {
        val today = LocalDate.of(2026, 10, 5)
        val cutoff = uploadCutoffs(today).adminClosedOnOrBefore
        assertEquals(LocalDate.of(2026, 10, 1), cutoff)
        assertTrue(lastUploadDate(cutoff, sharedRace = true).plusDays(1) <= today)
        assertTrue(lastUploadDate(cutoff.plusDays(1), sharedRace = true).plusDays(1) > today)
    }

    @Test fun `the owner cutoff is never early at a month end`() {
        val march31 = LocalDate.of(2028, 3, 31)
        assertTrue(LocalDate.of(2028, 2, 28) > uploadCutoffs(march31).ownerClosedOnOrBefore)

        var today = LocalDate.of(2028, 1, 1)
        while (today.year == 2028) {
            val cutoffs = uploadCutoffs(today)
            var d = cutoffs.ownerClosedOnOrBefore.minusDays(62)
            while (d <= cutoffs.ownerClosedOnOrBefore) {
                assertTrue(lastUploadDate(d, sharedRace = false).plusDays(1) <= today, "own event $d is still open on $today")
                d = d.plusDays(1)
            }
            assertTrue(lastUploadDate(cutoffs.adminClosedOnOrBefore, sharedRace = true).plusDays(1) <= today)
            today = today.plusDays(1)
        }
    }
}
