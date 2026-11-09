package com.quickpitik.service.photographer

import java.text.Normalizer

/**
 * Does a payout holder name belong to the person with this legal name (AA-I5)? Accents, case,
 * punctuation and initials are ignored. A match is the same words, or the shorter name (at least
 * two words) inside the longer one, so a middle name on either side still matches.
 */
object NameMatch {
    private val MARKS = Regex("\\p{M}+")
    private val NON_LETTERS = Regex("[^\\p{L}]+")

    fun same(legal: String, holder: String): Boolean {
        val (shorter, longer) = listOf(words(legal), words(holder)).sortedBy { it.size }
        return shorter.isNotEmpty() && (shorter == longer || (shorter.size >= 2 && longer.containsAll(shorter)))
    }

    private fun words(name: String): Set<String> =
        Normalizer.normalize(name, Normalizer.Form.NFD).replace(MARKS, "").lowercase()
            .split(NON_LETTERS).filter { it.length > 1 }.toSet()
}
