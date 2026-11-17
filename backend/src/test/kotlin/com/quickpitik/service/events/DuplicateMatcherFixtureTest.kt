package com.quickpitik.service.events

import org.junit.jupiter.api.Test
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertTrue

// Tunes and pins the duplicate cut-offs against labelled real race-name pairs
// (test resource races/cebu-race-pairs.csv). AA-R3 writes these values as the
// app.races.duplicate.* defaults.
class DuplicateMatcherFixtureTest {

    private val matcher = DuplicateMatcher(strongScore = STRONG, mediumScore = MEDIUM, dateToleranceDays = 1)

    private data class Pair(val a: RaceInput, val b: RaceCandidate, val same: Boolean)

    private fun pairs(): List<Pair> {
        val text = javaClass.getResource("/races/cebu-race-pairs.csv")!!.readText(Charsets.UTF_8)
        return text.lines()
            .filter { it.isNotBlank() && !it.startsWith("#") }
            .drop(1) // header
            .map { line ->
                val c = line.split(',')
                check(c.size == 9) { "bad fixture row: $line" }
                Pair(
                    a = RaceInput(c[0], LocalDate.parse(c[1]), c[2].ifBlank { null }, c[3].ifBlank { null }),
                    b = RaceCandidate(UUID.randomUUID(), c[4], LocalDate.parse(c[5]), c[6].ifBlank { null }, c[7].ifBlank { null }, emptySet()),
                    same = when (c[8]) {
                        "same" -> true
                        "different" -> false
                        else -> error("label must be same|different: $line")
                    },
                )
            }
    }

    @Test
    fun `no pair labelled different scores strong and every pair labelled same scores at least medium`() {
        val rows = pairs()
        assertTrue(rows.size >= 10, "the fixture needs at least 10 labelled pairs")

        val wrong = rows.mapNotNull { p ->
            val score = matcher.score(p.a, p.b)
            println("%.3f  %-9s  %s | %s".format(score, if (p.same) "same" else "different", p.a.name, p.b.name))
            when {
                !p.same && score >= STRONG -> "false join: ${p.a.name} | ${p.b.name} = $score"
                p.same && score < MEDIUM -> "missed: ${p.a.name} | ${p.b.name} = $score"
                else -> null
            }
        }
        assertTrue(wrong.isEmpty(), wrong.joinToString("\n"))
    }

    private companion object {
        // Tuned on the fixture (2026-11-08); see the AA-R2 journal.
        const val STRONG = 0.85
        const val MEDIUM = 0.5
    }
}
