package com.quickpitik.tools

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test

// RV-0.3: the percentile helper the calibration run prints its 5th, 50th and 95th from.
class RelevanceCalibrationMathTest {
    @Test
    fun `the middle of an odd list is its middle value, whatever the order`() {
        assertEquals(3.0, percentile(listOf(5.0, 1.0, 4.0, 2.0, 3.0), 50.0))
    }

    @Test
    fun `the 0th and 100th are the smallest and largest`() {
        val values = listOf(7.0, 2.0, 9.0, 4.0)
        assertEquals(2.0, percentile(values, 0.0))
        assertEquals(9.0, percentile(values, 100.0))
    }

    @Test
    fun `a percentile between two values is interpolated`() {
        // rank 0.25 * 3 = 0.75 between 1 and 2
        assertEquals(1.75, percentile(listOf(1.0, 2.0, 3.0, 4.0), 25.0)!!, 1e-9)
        // 5th of 0..100: rank 0.05 * 100 = 5
        assertEquals(5.0, percentile((0..100).map { it.toDouble() }, 5.0)!!, 1e-9)
    }

    @Test
    fun `one value is every percentile, and no values give no percentile`() {
        assertEquals(42.0, percentile(listOf(42.0), 95.0))
        assertNull(percentile(emptyList(), 50.0))
    }
}
