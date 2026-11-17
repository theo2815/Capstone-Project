package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.CaseBand
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RelevanceJunkRulesTest {

    private val p = RelevanceProperties(junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
    private val unset = RelevanceProperties()

    // A sharp, lit race photo; each test changes only what it is about.
    private val good = PhotoScores(
        fgSharpness = 80.0, fgBrightness = 60.0, sharpness = 70.0, brightness = 50.0,
        personConf = 0.99, raceConf = 0.9, measureError = null,
    )

    private fun counts(measured: Int, junk: Int) = SampleCounts(measured = measured, blurry = junk, dark = 0, notRace = 0, junk = junk)

    @Test fun `foreground sharpness under the cut-off is blurry`() {
        assertEquals(true, isBlurry(good.copy(fgSharpness = 19.9, sharpness = 90.0), p))
        assertEquals(false, isBlurry(good.copy(fgSharpness = 20.0, sharpness = 1.0), p))
    }

    @Test fun `with no foreground the whole image decides`() {
        assertEquals(true, isBlurry(good.copy(fgSharpness = null, sharpness = 10.0), p))
        assertEquals(false, isBlurry(good.copy(fgSharpness = null, sharpness = 30.0), p))
    }

    @Test fun `an unset cut-off answers unknown`() {
        assertNull(isBlurry(good, unset))
        assertNull(isDark(good, unset))
        assertNull(isJunk(good, unset))
        assertEquals(true, isJunk(good.copy(personConf = 0.0, raceConf = 0.0), unset))
    }

    @Test fun `brightness under the dark cut-off is dark`() {
        assertEquals(true, isDark(good.copy(brightness = 4.9), p))
        assertEquals(false, isDark(good.copy(brightness = 5.0), p))
    }

    @Test fun `a photo with a person is a race photo whatever else`() {
        assertEquals(false, isNotRace(good.copy(personConf = 0.6, raceConf = 0.0), p))
        assertEquals(false, isNotRace(good.copy(personConf = 0.0, raceConf = 0.6), p))
        assertEquals(true, isNotRace(good.copy(personConf = 0.59, raceConf = 0.59), p))
        assertEquals(false, isJunk(good.copy(raceConf = 0.0), p))
    }

    @Test fun `a blurry or dark photo is junk even with a person in it`() {
        assertEquals(true, isJunk(good.copy(fgSharpness = 5.0), p))
        assertEquals(true, isJunk(good.copy(brightness = 1.0), p))
        assertEquals(false, isJunk(good, p))
    }

    @Test fun `a photo that could not be measured is never junk`() {
        val bad = PhotoScores(null, null, null, null, null, null, measureError = "BAD_IMAGE")
        assertEquals(false, isJunk(bad, p))
        assertEquals(false, isJunk(bad.copy(fgSharpness = 1.0, brightness = 0.0, personConf = 0.0, raceConf = 0.0), p))
        assertNull(isJunk(PhotoScores(null, null, null, null, null, null, null), p))
    }

    @Test fun `a batch under the minimum size is small`() {
        assertEquals(CaseBand.SMALL, band(batchPhotos = 29, sampled = 29, counts = counts(29, 29), p = p))
        assertEquals(CaseBand.OVER, band(batchPhotos = 30, sampled = 30, counts = counts(30, 30), p = p))
    }

    @Test fun `under half the sample measured is unknown`() {
        assertEquals(CaseBand.UNKNOWN, band(batchPhotos = 1000, sampled = 300, counts = counts(149, 0), p = p))
        assertEquals(CaseBand.UNDER, band(batchPhotos = 1000, sampled = 300, counts = counts(150, 0), p = p))
        assertEquals(CaseBand.UNKNOWN, band(batchPhotos = 1000, sampled = 0, counts = counts(0, 0), p = p))
        assertEquals(CaseBand.UNKNOWN, band(batchPhotos = 1000, sampled = 300, counts = counts(300, 0), p = unset))
    }

    @Test fun `9 percent is under, 10 is mixed, 70 is mixed, 71 is over`() {
        fun at(junk: Int) = band(batchPhotos = 1000, sampled = 100, counts = counts(100, junk), p = p)
        assertEquals(CaseBand.UNDER, at(9))
        assertEquals(CaseBand.MIXED, at(10))
        assertEquals(CaseBand.MIXED, at(70))
        assertEquals(CaseBand.OVER, at(71))
    }

    @Test fun `65 to 75 percent is near the line`() {
        assertFalse(nearDumpLine(counts(100, 64), p))
        assertTrue(nearDumpLine(counts(100, 65), p))
        assertTrue(nearDumpLine(counts(100, 75), p))
        assertFalse(nearDumpLine(counts(100, 76), p))
        assertFalse(nearDumpLine(counts(0, 0), p))
    }

    @Test fun `sample counts leave out photos that could not be measured`() {
        val scores = listOf(
            good,
            good.copy(fgSharpness = 1.0),
            good.copy(brightness = 1.0),
            good.copy(fgSharpness = 1.0, brightness = 1.0),
            good.copy(personConf = 0.1, raceConf = 0.1),
            good.copy(fgSharpness = 1.0, measureError = "GAVE_UP"),
            PhotoScores(null, null, null, null, null, null, measureError = null),
        )
        assertEquals(SampleCounts(measured = 5, blurry = 2, dark = 2, notRace = 1, junk = 4), sampleCounts(scores, p))
        assertEquals(SampleCounts(measured = 5, blurry = 0, dark = 0, notRace = 1, junk = 1), sampleCounts(scores, unset),
            "with the cut-offs unset only the label rule counts")
    }
}
