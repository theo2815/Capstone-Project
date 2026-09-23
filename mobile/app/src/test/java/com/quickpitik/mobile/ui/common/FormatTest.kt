package com.quickpitik.mobile.ui.common

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import java.util.Locale
import java.util.TimeZone

/**
 * Money and backend timestamps must read the same on every phone. The screens
 * formatted with the phone's region (a German phone showed ₱1.234,50) and read
 * a UTC timestamp's date straight off the string, so a 3 AM Manila sale landed
 * on the previous day and the receipt, the earnings list and the website could
 * disagree about the same instant. The JVM is set to a German, New York phone.
 */
class FormatTest {
    private lateinit var locale: Locale
    private lateinit var zone: TimeZone

    @Before fun phoneAbroad() {
        locale = Locale.getDefault()
        zone = TimeZone.getDefault()
        Locale.setDefault(Locale.GERMANY)
        TimeZone.setDefault(TimeZone.getTimeZone("America/New_York"))
    }

    @After fun restore() {
        Locale.setDefault(locale)
        TimeZone.setDefault(zone)
    }

    @Test fun `pesos keep the website's grouping and round half up`() {
        assertEquals("₱1,234.50", peso(1234.5))
        assertEquals("₱1,235", peso(1234.5, 0))
        assertEquals("₱1,234,567.89", peso(1234567.891))
    }

    @Test fun `counts group like the website, not the phone's region`() {
        // The phone is set to Germany in @Before, where "%,d".format(1234) reads "1.234".
        assertEquals("1,234", count(1234))
        assertEquals("1,234,567", count(1234567))
    }

    @Test fun `backend timestamps read on Manila's clock`() {
        // 20:00 UTC on Aug 31 is 04:00 on Sep 1 in Manila: a new day and a new month.
        val pattern = "MMM d, yyyy · h:mm a"
        assertEquals("Sep 1, 2026 · 4:00 AM", formatManila("2026-08-31T20:00:00Z", pattern))
        assertEquals("Sep 1, 2026 · 4:00 AM", formatManila("2026-09-01T04:00:00+08:00", pattern))
        assertEquals("2026-09", formatManila("2026-08-31T20:00:00.123456Z", "yyyy-MM"))
        assertEquals("SEP 1, 2026", formatInboxDate("2026-08-31T20:00:00Z"))
    }

    @Test fun `a missing or unreadable timestamp gives null so each screen keeps its placeholder`() {
        for (iso in listOf(null, "", "not a date")) {
            assertNull(iso, formatManila(iso, "yyyy-MM-dd"))
        }
        assertEquals("not a date", formatInboxDate("not a date"))
    }
}
