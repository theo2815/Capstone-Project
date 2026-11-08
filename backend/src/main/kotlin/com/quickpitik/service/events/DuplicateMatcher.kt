package com.quickpitik.service.events

import java.text.Normalizer
import java.time.LocalDate
import java.time.temporal.ChronoUnit
import java.util.UUID
import kotlin.math.abs

data class RaceInput(
    val name: String,
    val date: LocalDate,
    val provinceCode: String?,
    val organizerName: String?,
)

data class RaceCandidate(
    val id: UUID,
    val name: String,
    val date: LocalDate,
    val provinceCode: String?,
    val organizerName: String?,
    /** nameKey() of each known alias of this race. */
    val aliasKeys: Set<String>,
)

data class Scored(val candidate: RaceCandidate, val score: Double)

/** Candidates at or above the strong cut-off, and the medium ones below it; best first. */
data class DuplicateResult(val strong: List<Scored>, val medium: List<Scored>)

/**
 * Scores a typed race against existing races to catch two photographers creating
 * the same real race. Pure: no Spring, no database; the caller (AA-R3) loads the
 * candidates within the date tolerance and passes the cut-offs from config.
 */
class DuplicateMatcher(
    private val strongScore: Double,
    private val mediumScore: Double,
    private val dateToleranceDays: Long,
) {

    /** A stable key: folded, lowercased, years / ordinals / distances / filler dropped, distinct words sorted. */
    fun nameKey(name: String): String = keyTokens(name).sorted().joinToString(" ")

    /** 0.0 when the dates are outside the tolerance; otherwise name similarity with the adjustments. */
    fun score(input: RaceInput, candidate: RaceCandidate): Double {
        val daysApart = abs(ChronoUnit.DAYS.between(input.date, candidate.date))
        if (daysApart > dateToleranceDays) return 0.0

        val inputTokens = keyTokens(input.name)
        var score = (setOf(nameKey(candidate.name)) + candidate.aliasKeys)
            .maxOf { jaccard(inputTokens, it.split(' ').filter(String::isNotEmpty).toSet()) }

        if (!input.organizerName.isNullOrBlank() && !candidate.organizerName.isNullOrBlank() &&
            jaccard(keyTokens(input.organizerName), keyTokens(candidate.organizerName)) >= ORGANIZER_OVERLAP
        ) {
            score += ORGANIZER_BONUS
        }
        if (daysApart >= 1) score -= ONE_DAY_PENALTY
        score = score.coerceIn(0.0, 1.0)
        // Two known, different provinces: never strong, at most an admin decision.
        if (input.provinceCode != null && candidate.provinceCode != null && input.provinceCode != candidate.provinceCode) {
            score = minOf(score, Math.nextDown(strongScore))
        }
        return score
    }

    fun classify(input: RaceInput, candidates: List<RaceCandidate>): DuplicateResult {
        val scored = candidates.map { Scored(it, score(input, it)) }.sortedByDescending { it.score }
        return DuplicateResult(
            strong = scored.filter { it.score >= strongScore },
            medium = scored.filter { it.score >= mediumScore && it.score < strongScore },
        )
    }

    private fun keyTokens(name: String): Set<String> =
        Normalizer.normalize(name, Normalizer.Form.NFKD)
            .replace(MARKS, "")
            .lowercase()
            .replace("&", " and ")
            .replace(NON_ALNUM, " ")
            .split(' ')
            .filter { it.isNotEmpty() && it !in FILLER && !DROPPED.matches(it) }
            .toSet()

    private fun jaccard(a: Set<String>, b: Set<String>): Double {
        val union = a union b
        return if (union.isEmpty()) 0.0 else (a intersect b).size.toDouble() / union.size
    }

    private companion object {
        val MARKS = Regex("\\p{M}+")
        val NON_ALNUM = Regex("[^a-z0-9]+")
        // Years, ordinals and distances: "2027", "3rd", "42k", "21km", "10mi".
        val DROPPED = Regex("(19|20)\\d\\d|\\d+(st|nd|rd|th)|\\d+(k|km|mi)")
        val FILLER = setOf(
            "the", "a", "an", "of", "and", "at", "in", "sa", "ng",
            "annual", "edition", "official", "presents", "presented", "by",
        )
        const val ORGANIZER_OVERLAP = 0.5
        const val ORGANIZER_BONUS = 0.1
        const val ONE_DAY_PENALTY = 0.05
    }
}
