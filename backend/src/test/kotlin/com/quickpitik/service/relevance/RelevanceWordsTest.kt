package com.quickpitik.service.relevance

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class RelevanceWordsTest {

    @Test fun `words with a digit are dropped`() {
        assertEquals(listOf("CEBU"), cleanWords(listOf("1234", "CEBU", "KM21", "2003", "A1B")))
    }

    @Test fun `words are upper-cased and counted once per photo`() {
        assertEquals(listOf("CEBU", "MARATHON"), cleanWords(listOf("Cebu", "marathon", "CEBU", "cebu", "Marathon")))
    }

    @Test fun `words under 3 or over 40 letters are dropped`() {
        val forty = "A".repeat(40)
        // A hyphen is not a letter; a letter with an accent is.
        assertEquals(listOf("RUN", forty, "ÑAN"), cleanWords(listOf("GO", "RUN", forty, "B".repeat(41), "", "TO-GO", "ñan")))
    }

    @Test fun `at most 40 words are kept`() {
        val raw = List(45) { i -> "W" + ('A' + i / 26) + ('A' + i % 26) }
        val kept = cleanWords(raw)
        assertEquals(40, kept.size)
        assertEquals(raw.take(40), kept, "in the order read")
    }
}
