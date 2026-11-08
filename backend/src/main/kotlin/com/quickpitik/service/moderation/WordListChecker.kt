package com.quickpitik.service.moderation

import org.springframework.beans.factory.annotation.Autowired
import org.springframework.core.io.ClassPathResource
import org.springframework.stereotype.Component
import java.text.Normalizer
import java.util.Locale

enum class WordList(vararg val files: String) {
    BAD_WORDS("bad-words-en.txt", "bad-words-tl.txt", "bad-words-ceb.txt"),
    RACE_WORDS("race-words.txt"),
    NON_RACE_WORDS("non-race-words.txt"),
}

// Word lists for profile text (bad words) and race names (race / non-race words).
// shortcut: whole-token match only (no leetspeak or joined-word folding), add folding for BAD_WORDS if admins see bypasses
@Component
class WordListChecker internal constructor(dir: String) {

    @Autowired
    constructor() : this("moderation")

    // Each entry is kept as its normalised token list, so "fun run" matches two consecutive tokens.
    private val entries: Map<WordList, List<List<String>>> = WordList.entries.associateWith { list ->
        list.files.flatMap { file -> load("$dir/$file") }
    }

    /** Distinct normalised entries found in [text]; empty = clean. */
    fun hits(text: String, lists: Set<WordList>): List<String> {
        val tokens = tokens(text)
        return lists.flatMap { entries.getValue(it) }
            .filter { entry -> tokens.windowed(entry.size).any { it == entry } }
            .map { it.joinToString(" ") }
            .distinct()
    }

    private fun load(path: String): List<List<String>> {
        val resource = ClassPathResource(path)
        check(resource.exists()) { "word list $path is missing" }
        val words = resource.inputStream.bufferedReader().readLines()
            .map { it.substringBefore('#') }
            .map { tokens(it) }
            .filter { it.isNotEmpty() }
        check(words.isNotEmpty()) { "word list $path is empty" }
        return words
    }

    private fun tokens(text: String): List<String> =
        Normalizer.normalize(text.lowercase(Locale.ROOT), Normalizer.Form.NFD)
            .replace(COMBINING_MARKS, "")
            .split(NON_WORD)
            .filter { it.isNotEmpty() }

    private companion object {
        val COMBINING_MARKS = Regex("\\p{M}+")
        val NON_WORD = Regex("[^\\p{L}\\p{N}]+")
    }
}
