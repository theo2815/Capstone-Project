package com.quickpitik.config

import org.junit.jupiter.api.Test
import org.springframework.boot.context.properties.bind.BindException
import org.springframework.boot.context.properties.bind.Binder
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource
import org.springframework.boot.env.YamlPropertySourceLoader
import org.springframework.core.env.StandardEnvironment
import org.springframework.core.io.ClassPathResource
import java.time.Duration
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RelevancePropertiesTest {

    @Test fun `defaults are off with the placeholder thresholds`() {
        val p = RelevanceProperties()
        assertEquals(RelevanceMode.OFF, p.mode)
        assertEquals(60_000L, p.sweepIntervalMs)
        assertEquals(10L, p.quietMinutes)
        assertEquals(100, p.lookupsPerTick)
        assertEquals(5, p.maxBatchesPerTick)
        assertEquals(60L, p.rescoreMinutes)
        assertEquals(7L, p.settleDays)
        assertEquals(Duration.ofSeconds(5), p.lookupTimeout)
        assertEquals(RelevanceProperties.Copy(tightDistance = 4, weakDistance = 8, minHits = 3, minHitSharePercent = 5), p.copy)
        assertEquals(RelevanceProperties.Date(windowDays = 2, minDatedPhotos = 10, outsideSharePercent = 80), p.date)
        assertEquals(RelevanceProperties.Junk(blurCutoff = null, darkCutoff = null, labelMinConfidence = 0.6,
            raceLabels = listOf("Running", "Marathon", "Road"), fitsBelowPercent = 10, dumpAbovePercent = 70), p.junk)
        assertEquals(RelevanceProperties.Check(minBatchSize = 30, nearLinePoints = 5), p.check)
    }

    @Test fun `the yml binds every key and names its env override`() {
        val env = StandardEnvironment()
        YamlPropertySourceLoader().load("yml", ClassPathResource("application.yml")).forEach { env.propertySources.addLast(it) }
        val bound = Binder.get(env).bind("app.relevance", RelevanceProperties::class.java).get()
        assertEquals(RelevanceProperties(), bound)

        val yml = javaClass.getResource("/application.yml")!!.readText()
        listOf(
            "RELEVANCE_MODE:off", "RELEVANCE_SWEEP_INTERVAL_MS:60000", "RELEVANCE_QUIET_MINUTES:10",
            "RELEVANCE_LOOKUPS_PER_TICK:100", "RELEVANCE_MAX_BATCHES_PER_TICK:5", "RELEVANCE_RESCORE_MINUTES:60",
            "RELEVANCE_SETTLE_DAYS:7", "RELEVANCE_LOOKUP_TIMEOUT:5s", "RELEVANCE_COPY_TIGHT_DISTANCE:4", "RELEVANCE_COPY_WEAK_DISTANCE:8",
            "RELEVANCE_COPY_MIN_HITS:3", "RELEVANCE_COPY_MIN_HIT_SHARE_PERCENT:5", "RELEVANCE_DATE_WINDOW_DAYS:2",
            "RELEVANCE_DATE_MIN_DATED_PHOTOS:10", "RELEVANCE_DATE_OUTSIDE_SHARE_PERCENT:80",
            "RELEVANCE_JUNK_BLUR_CUTOFF:}", "RELEVANCE_JUNK_DARK_CUTOFF:}",
            "RELEVANCE_JUNK_LABEL_MIN_CONFIDENCE:0.6", "RELEVANCE_JUNK_RACE_LABELS:Running,Marathon,Road",
            "RELEVANCE_JUNK_FITS_BELOW_PERCENT:10", "RELEVANCE_JUNK_DUMP_ABOVE_PERCENT:70",
            "RELEVANCE_CHECK_MIN_BATCH_SIZE:30", "RELEVANCE_CHECK_NEAR_LINE_POINTS:5",
        ).forEach { assertTrue(it in yml, "missing $it in application.yml") }
    }

    private fun bind(vararg pairs: Pair<String, String>): RelevanceProperties =
        Binder(MapConfigurationPropertySource(pairs.associate { "app.relevance.${it.first}" to it.second }))
            .bindOrCreate("app.relevance", RelevanceProperties::class.java)

    @Test fun `record-only and auto-hide bind`() {
        assertEquals(RelevanceMode.RECORD_ONLY, bind("mode" to "record-only").mode)
        assertEquals(RelevanceMode.AUTO_HIDE, bind("mode" to "auto-hide").mode)
        assertEquals(RelevanceMode.OFF, bind("mode" to "off").mode)
    }

    @Test fun `the old names shadow and flag still bind`() {
        assertEquals(RelevanceMode.SHADOW, bind("mode" to "shadow").mode)
        assertEquals(RelevanceMode.FLAG, bind("mode" to "flag").mode)
    }

    @Test fun `an unknown mode stops the boot`() {
        assertFailsWith<BindException> { bind("mode" to "hide-everything") }
    }

    @Test fun `an empty cut-off binds as not set`() {
        val p = bind("junk.blur-cutoff" to "", "junk.dark-cutoff" to "")
        assertNull(p.junk.blurCutoff)
        assertNull(p.junk.darkCutoff)
        assertEquals(RelevanceProperties.Junk(blurCutoff = 42.5, darkCutoff = 0.0),
            bind("junk.blur-cutoff" to "42.5", "junk.dark-cutoff" to "0").junk)
    }

    @Test fun `a cut-off above 100 is refused`() {
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Junk(blurCutoff = 100.1) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Junk(darkCutoff = 101.0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Junk(blurCutoff = -0.1) }
        assertFailsWith<BindException> { bind("junk.dark-cutoff" to "250") }
    }

    @Test fun `junk shares out of order, a confidence above one or no race labels are refused`() {
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Junk(fitsBelowPercent = 71, dumpAbovePercent = 70) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Junk(fitsBelowPercent = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Junk(dumpAbovePercent = 100) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Junk(labelMinConfidence = 60.0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Junk(raceLabels = emptyList()) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Check(minBatchSize = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Check(nearLinePoints = -1) }
    }

    @Test fun `a tight distance above the weak distance is refused`() {
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Copy(tightDistance = 9, weakDistance = 8) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Copy(tightDistance = -1) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Copy(weakDistance = 65) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Copy(minHits = 0) }
    }

    @Test fun `a share percent outside one to a hundred is refused`() {
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Copy(minHitSharePercent = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Copy(minHitSharePercent = 101) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Date(outsideSharePercent = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Date(outsideSharePercent = 101) }
    }

    @Test fun `intervals and tick counts below one, or too many look-ups, are refused`() {
        assertFailsWith<IllegalArgumentException> { RelevanceProperties(sweepIntervalMs = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties(quietMinutes = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties(lookupsPerTick = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties(lookupsPerTick = 1001) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties(maxBatchesPerTick = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties(rescoreMinutes = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties(settleDays = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties(lookupTimeout = Duration.ZERO) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Date(windowDays = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Date(minDatedPhotos = 0) }
    }
}
