package com.quickpitik.mobile.ui.runner

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** The only place a typed bib is normalised before it reaches the search. */
class RunnerGalleryInputTest {
    @Test fun `a typed bib is trimmed and upper-cased, and blank means no search`() {
        assertEquals("B-4082", normalizeBib(" b-4082 "))
        assertEquals("1234", normalizeBib("1234"))
        assertNull(normalizeBib("   "))
        assertNull(normalizeBib(""))
    }
}
