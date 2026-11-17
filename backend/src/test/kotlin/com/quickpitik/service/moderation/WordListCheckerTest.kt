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
        // Names that look like listed words must stay clean.
        assertTrue(checker.hits("Dick Reyes, Lady Gaga fan", setOf(WordList.BAD_WORDS)).isEmpty())
    }

    @Test
    fun `a listed word with one Cyrillic look-alike letter still hits`() {
        // second letter is Cyrillic a (U+0430), not Latin a
        val word = "g\u0430go"
        assertEquals(listOf(word), checker.hits(word, setOf(WordList.BAD_WORDS)))
    }

    @Test
    fun `a name written wholly in another script has no hit`() {
        assertTrue(checker.hits("\u041c\u0430\u0440\u0430\u0444\u043e\u043d \ud55c\uad6d \u4e2d\u6587", WordList.entries.toSet()).isEmpty())
    }

    @Test
    fun `a mixed-script word is not a race word`() {
        // "marathon" with a Cyrillic a is not folded into the race or non-race lists
        assertTrue(checker.hits("m\u0430rathon", setOf(WordList.RACE_WORDS)).isEmpty())
        assertTrue(checker.hits("m\u0430rathon", setOf(WordList.NON_RACE_WORDS)).isEmpty())
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
