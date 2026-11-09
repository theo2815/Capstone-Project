package com.quickpitik.service.photographer

import org.junit.jupiter.api.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class NameMatchTest {
    @Test
    fun `a middle initial, capitals and dots still match`() = assertTrue(NameMatch.same("Juan Dela Cruz", "JUAN A. DELA CRUZ"))

    @Test
    fun `accents are ignored`() = assertTrue(NameMatch.same("José Rizal", "Jose Rizal"))

    @Test
    fun `a middle name on one side only still matches`() = assertTrue(NameMatch.same("Juan Cruz", "Juan Pablo Cruz"))

    @Test
    fun `a first name alone is not enough`() = assertFalse(NameMatch.same("Juan", "Juan Cruz"))

    @Test
    fun `a different first name is a mismatch`() = assertFalse(NameMatch.same("Maria Dela Cruz", "Juan Dela Cruz"))
}
