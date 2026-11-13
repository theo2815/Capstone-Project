package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.CaseBand
import kotlin.math.abs

// RV-1.4 (Photo Relevance v2, G4, G3, G5): what makes a photo junk and which band a sample falls in.
// Pure and read from stored raw scores, so a new cut-off re-classes old photos at no AWS cost.
// null means unknown: a cut-off is unset, or the score was never measured.

// Scores 0-100, confidences 0-1. measureError set = the photo could not be measured.
data class PhotoScores(
    val fgSharpness: Double?, val fgBrightness: Double?, val sharpness: Double?, val brightness: Double?,
    val personConf: Double?, val raceConf: Double?, val measureError: String?,
)

// measured excludes photos with a measure error; the other counts are within measured.
data class SampleCounts(val measured: Int, val blurry: Int, val dark: Int, val notRace: Int, val junk: Int)

// Foreground first; the whole image only when Rekognition found no foreground (P11).
internal fun isBlurry(s: PhotoScores, p: RelevanceProperties): Boolean? {
    val cutoff = p.junk.blurCutoff ?: return null
    return (s.fgSharpness ?: s.sharpness ?: return null) < cutoff
}

internal fun isDark(s: PhotoScores, p: RelevanceProperties): Boolean? {
    val cutoff = p.junk.darkCutoff ?: return null
    return (s.brightness ?: return null) < cutoff
}

// Narrow on purpose (R9): a person, or any race label, makes it a race photo.
internal fun isNotRace(s: PhotoScores, p: RelevanceProperties): Boolean? {
    val person = s.personConf ?: return null
    val race = s.raceConf ?: return null
    return person < p.junk.labelMinConfidence && race < p.junk.labelMinConfidence
}

// Any rule true is junk; else any unknown is unknown. A photo that could not be measured is never junk.
internal fun isJunk(s: PhotoScores, p: RelevanceProperties): Boolean? {
    if (s.measureError != null) return false
    val rules = listOf(isBlurry(s, p), isDark(s, p), isNotRace(s, p))
    return when {
        true in rules -> true
        null in rules -> null
        else -> false
    }
}

internal fun band(batchPhotos: Int, sampled: Int, counts: SampleCounts, p: RelevanceProperties): CaseBand = when {
    batchPhotos < p.check.minBatchSize -> CaseBand.SMALL
    p.junk.blurCutoff == null || p.junk.darkCutoff == null -> CaseBand.UNKNOWN
    counts.measured == 0 || counts.measured * 2 < sampled -> CaseBand.UNKNOWN
    counts.junk * 100 < p.junk.fitsBelowPercent * counts.measured -> CaseBand.UNDER
    counts.junk * 100 > p.junk.dumpAbovePercent * counts.measured -> CaseBand.OVER
    else -> CaseBand.MIXED
}

// A photo with a measure error, or never measured (no label confidence stored), is left out.
internal fun sampleCounts(scores: List<PhotoScores>, p: RelevanceProperties): SampleCounts {
    val measured = scores.filter { it.measureError == null && it.personConf != null }
    return SampleCounts(
        measured = measured.size,
        blurry = measured.count { isBlurry(it, p) == true },
        dark = measured.count { isDark(it, p) == true },
        notRace = measured.count { isNotRace(it, p) == true },
        junk = measured.count { isJunk(it, p) == true },
    )
}

// The share is within nearLinePoints of the dump line: take a second sample (G5).
internal fun nearDumpLine(counts: SampleCounts, p: RelevanceProperties): Boolean =
    counts.measured > 0 &&
        abs(counts.junk * 100 - p.junk.dumpAbovePercent * counts.measured) <= p.check.nearLinePoints * counts.measured
