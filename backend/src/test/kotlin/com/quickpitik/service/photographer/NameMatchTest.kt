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

    private val card = listOf("REPUBLIC OF THE PHILIPPINES", "DELA CRUZ, JUAN ANDRES")

    @Test
    fun `a legal name is on the card in any order and without its middle name`() = assertTrue(NameMatch.onCard("Juan Dela Cruz", card))

    @Test
    fun `a legal name is on the card with accents and initials ignored`() =
        assertTrue(NameMatch.onCard("José Rizal", listOf("RIZAL JOSE P.")))

    @Test
    fun `a legal name word the card does not print is not on the card`() = assertFalse(NameMatch.onCard("Maria Dela Cruz", card))

    @Test
    fun `a one-word legal name is never on the card`() = assertFalse(NameMatch.onCard("Juan", card))

    @Test
    fun `no lines hold no name`() = assertFalse(NameMatch.onCard("Juan Dela Cruz", emptyList()))
}
