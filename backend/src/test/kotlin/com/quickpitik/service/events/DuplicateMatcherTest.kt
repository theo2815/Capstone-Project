package com.quickpitik.service.events

import org.junit.jupiter.api.Test
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DuplicateMatcherTest {

    private val matcher = DuplicateMatcher(strongScore = 0.85, mediumScore = 0.5, dateToleranceDays = 1)
    private val day = LocalDate.of(2027, 1, 10)

    private fun candidate(name: String, date: LocalDate = day, province: String? = null, aliases: Set<String> = emptySet()) =
        RaceCandidate(id = UUID.randomUUID(), name = name, date = date, provinceCode = province, organizerName = null, aliasKeys = aliases)

    @Test
    fun `name key drops case punctuation year distance and filler`() {
        assertEquals("cebu city marathon", matcher.nameKey("The 2027 Cebu City Marathon: 42K & 21km, 3rd Annual Edition!"))
        assertEquals("cebu marathon", matcher.nameKey("CEBU  Marathon 2027"))
        // Accents fold, hyphens split, words come out sorted.
        assertEquals("cebu run sun", matcher.nameKey("Sún Run — Cebu"))
        assertEquals("half marathon srp", matcher.nameKey("SRP Half-Marathon"))
    }

    @Test
    fun `province conflict never scores strong`() {
        val input = RaceInput("Bohol Trail Classic", day, provinceCode = "bohol", organizerName = null)
        val result = matcher.classify(input, listOf(candidate("Bohol Trail Classic", province = "cebu")))

        assertTrue(result.strong.isEmpty())
        assertEquals(1, result.medium.size)
        assertTrue(matcher.score(input, candidate("Bohol Trail Classic", province = "bohol")) >= 0.85)
    }

    @Test
    fun `two days apart is no candidate`() {
        val input = RaceInput("Cebu City Marathon", day, provinceCode = null, organizerName = null)
        val far = candidate("Cebu City Marathon", date = day.plusDays(2))

        assertEquals(0.0, matcher.score(input, far))
        val result = matcher.classify(input, listOf(far))
        assertTrue(result.strong.isEmpty() && result.medium.isEmpty())
        // One day apart still counts, a little lower than the same day.
        assertTrue(matcher.score(input, candidate("Cebu City Marathon", date = day.minusDays(1))) < matcher.score(input, candidate("Cebu City Marathon")))
        assertTrue(matcher.classify(input, listOf(candidate("Cebu City Marathon", date = day.minusDays(1)))).strong.isNotEmpty())
    }

    @Test
    fun `alias match scores like the name`() {
        val input = RaceInput("Sinulog Festival Fun Run", day, provinceCode = null, organizerName = null)
        val withAlias = candidate("Sinulog Run", aliases = setOf(matcher.nameKey("Sinulog Festival Fun Run")))

        assertEquals(matcher.score(input, candidate("Sinulog Festival Fun Run")), matcher.score(input, withAlias))
        assertEquals(1.0, matcher.score(input, withAlias))
    }

    @Test
    fun `a shared organizer lifts a medium name match`() {
        val input = RaceInput("Cebu Night Run", day, provinceCode = null, organizerName = "CCM Organizing Committee")
        val noOrganizer = matcher.score(input, candidate("Cebu City Night Run"))
        val sameOrganizer = matcher.score(input, candidate("Cebu City Night Run").copy(organizerName = "CCM Organizing Committee"))

        assertEquals(noOrganizer + 0.1, sameOrganizer, 1e-9)
    }

    @Test
    fun `results come best first`() {
        val input = RaceInput("Cebu City Marathon", day, provinceCode = null, organizerName = null)
        val exact = candidate("Cebu City Marathon")
        val close = candidate("Cebu City Marathon", date = day.plusDays(1))
        val medium = candidate("Cebu Marathon")

        val result = matcher.classify(input, listOf(close, medium, exact))

        assertEquals(listOf(exact.id, close.id), result.strong.map { it.candidate.id })
        assertEquals(listOf(medium.id), result.medium.map { it.candidate.id })
    }
}
