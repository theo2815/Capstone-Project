package com.quickpitik.service.moderation

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// Real list files from src/main/resources/moderation, no Spring context.
class WordListCheckerTest {

    private val checker = WordListChecker()

    @Test
    fun `finds a bad word whatever the case and accents`() {
        assertEquals(listOf("gago"), checker.hits("Gágo Photos", setOf(WordList.BAD_WORDS)))
        assertEquals(listOf("shit"), checker.hits("SHIT happens", setOf(WordList.BAD_WORDS)))
    }

    @Test
    fun `does not match inside a longer innocent word`() {
        // "puta" is listed; "computation" and "reputable" must stay clean.
        assertTrue(checker.hits("computation reputable photography", setOf(WordList.BAD_WORDS)).isEmpty())
    }

    @Test
    fun `matches a two-word entry across spaces and hyphens`() {
        // "run" is a race word on its own too, so check that the pair is among the hits.
        assertTrue(checker.hits("Cebu Fun-Run 2027", setOf(WordList.RACE_WORDS)).contains("fun run"))
        assertTrue(checker.hits("cebu  fun   run", setOf(WordList.RACE_WORDS)).contains("fun run"))
        assertEquals(listOf("tang ina"), checker.hits("tang-ina", setOf(WordList.BAD_WORDS)))
        assertTrue(checker.hits("fun times run", setOf(WordList.RACE_WORDS)).none { it == "fun run" })
    }

    @Test
    fun `21k and fun run are race words`() {
        assertTrue(checker.hits("Cebu City 21K", setOf(WordList.RACE_WORDS)).contains("21k"))
        assertTrue(checker.hits("Mandaue Fun Run", setOf(WordList.RACE_WORDS)).contains("fun run"))
    }

    @Test
    fun `triathlon is a non-race word`() {
        assertEquals(listOf("triathlon"), checker.hits("Cebu Triathlon", setOf(WordList.NON_RACE_WORDS)))
        assertTrue(checker.hits("Cebu Triathlon", setOf(WordList.RACE_WORDS)).isEmpty())
    }

    @Test
    fun `clean text has no hits in any list`() {
        assertTrue(checker.hits("Juan Dela Cruz Photography", WordList.entries.toSet()).isEmpty())
    }

    @Test
    fun `every list file loads and is non-empty`() {
        // One known entry from each of the five files (an empty file fails construction).
        assertEquals(listOf("fuck"), checker.hits("fuck", setOf(WordList.BAD_WORDS))) // en
        assertEquals(listOf("gago"), checker.hits("gago", setOf(WordList.BAD_WORDS))) // tl
        assertEquals(listOf("yawa"), checker.hits("yawa", setOf(WordList.BAD_WORDS))) // ceb
        assertEquals(listOf("marathon"), checker.hits("marathon", setOf(WordList.RACE_WORDS)))
        assertEquals(listOf("wedding"), checker.hits("wedding", setOf(WordList.NON_RACE_WORDS)))
    }

    @Test
    fun `a missing list file fails at construction`() {
        assertFailsWith<IllegalStateException> { WordListChecker("moderation-missing") }
    }
}
