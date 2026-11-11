package com.quickpitik.config

import org.junit.jupiter.api.Test
import org.springframework.boot.context.properties.bind.Binder
import org.springframework.boot.env.YamlPropertySourceLoader
import org.springframework.core.env.StandardEnvironment
import org.springframework.core.io.ClassPathResource
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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
        assertEquals(RelevanceProperties.Copy(tightDistance = 4, weakDistance = 8, minHits = 3, minHitSharePercent = 5), p.copy)
        assertEquals(RelevanceProperties.Date(windowDays = 2, minDatedPhotos = 10, outsideSharePercent = 80), p.date)
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
            "RELEVANCE_SETTLE_DAYS:7", "RELEVANCE_COPY_TIGHT_DISTANCE:4", "RELEVANCE_COPY_WEAK_DISTANCE:8",
            "RELEVANCE_COPY_MIN_HITS:3", "RELEVANCE_COPY_MIN_HIT_SHARE_PERCENT:5", "RELEVANCE_DATE_WINDOW_DAYS:2",
            "RELEVANCE_DATE_MIN_DATED_PHOTOS:10", "RELEVANCE_DATE_OUTSIDE_SHARE_PERCENT:80",
        ).forEach { assertTrue(it in yml, "missing $it in application.yml") }
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
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Date(windowDays = 0) }
        assertFailsWith<IllegalArgumentException> { RelevanceProperties.Date(minDatedPhotos = 0) }
    }
}
