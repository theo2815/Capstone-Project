package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

// PE-1: photo-to-event relevance. The sweep only writes evidence; a person decides.
// Thresholds are placeholders until PE-0 measures them (PE-1.7).
@ConfigurationProperties(prefix = "app.relevance")
data class RelevanceProperties(
    // Kill switch. Read it through RelevanceSwitch, which turns it into what may run now.
    val mode: RelevanceMode = RelevanceMode.OFF,
    // Tick cadence, and the first delay after boot.
    val sweepIntervalMs: Long = 60_000,
    // A batch is due this long after its newest upload.
    val quietMinutes: Long = 10,
    // Look-ups per tick, all batches together.
    val lookupsPerTick: Int = 100,
    val maxBatchesPerTick: Int = 5,
    // Wait before re-scoring a batch that was unsettled at its last scoring.
    val rescoreMinutes: Long = 60,
    // After uploads close, stop waiting for late EXIF after this.
    val settleDays: Long = 7,
    // Statement timeout of one look-up transaction; a slow event fails that batch's step, not the tick.
    val lookupTimeout: Duration = Duration.ofSeconds(5),
    val copy: Copy = Copy(),
    val date: Date = Date(),
    val junk: Junk = Junk(),
    val check: Check = Check(),
) {
    init {
        require(sweepIntervalMs >= 1) { "sweepIntervalMs must be at least 1" }
        require(quietMinutes >= 1) { "quietMinutes must be at least 1" }
        require(lookupsPerTick in 1..1000) { "lookupsPerTick must be 1..1000" }
        require(maxBatchesPerTick >= 1) { "maxBatchesPerTick must be at least 1" }
        require(rescoreMinutes >= 1) { "rescoreMinutes must be at least 1" }
        require(settleDays >= 1) { "settleDays must be at least 1" }
        require(lookupTimeout >= Duration.ofMillis(1)) { "lookupTimeout must be at least 1ms" }
    }

    // D20: a hit is an earlier photo by another photographer within tightDistance bits.
    data class Copy(
        val tightDistance: Int = 4,
        val weakDistance: Int = 8,
        val minHits: Int = 3,
        val minHitSharePercent: Int = 5,
    ) {
        init {
            require(tightDistance in 0..weakDistance && weakDistance <= 64) { "need 0 <= tightDistance <= weakDistance <= 64" }
            require(minHits >= 1) { "minHits must be at least 1" }
            require(minHitSharePercent in 1..100) { "minHitSharePercent must be 1..100" }
        }
    }

    // D21: capture dates outside the event date +/- windowDays.
    data class Date(
        val windowDays: Long = 2,
        val minDatedPhotos: Int = 10,
        val outsideSharePercent: Int = 80,
    ) {
        init {
            require(windowDays >= 1) { "windowDays must be at least 1" }
            require(minDatedPhotos >= 1) { "minDatedPhotos must be at least 1" }
            require(outsideSharePercent in 1..100) { "outsideSharePercent must be 1..100" }
        }
    }

    // Version 2 junk cut-offs (0-100 scores). No default until the calibration run (O7); while either is
    // unset, auto-hide runs as record-only.
    data class Junk(
        val blurCutoff: Double? = null,
        val darkCutoff: Double? = null,
        // Not a race: neither Person nor any race label reaches this confidence (0-1).
        val labelMinConfidence: Double = 0.6,
        val raceLabels: List<String> = listOf("Running", "Marathon", "Road"),
        // Junk share of the measured sample: under fitsBelowPercent is UNDER, over dumpAbovePercent is OVER.
        val fitsBelowPercent: Int = 10,
        val dumpAbovePercent: Int = 70,
    ) {
        init {
            require(blurCutoff == null || blurCutoff in 0.0..100.0) { "blurCutoff must be 0..100" }
            require(darkCutoff == null || darkCutoff in 0.0..100.0) { "darkCutoff must be 0..100" }
            require(labelMinConfidence in 0.0..1.0) { "labelMinConfidence must be 0..1" }
            require(raceLabels.isNotEmpty()) { "raceLabels must not be empty" }
            require(fitsBelowPercent in 1..dumpAbovePercent && dumpAbovePercent < 100) {
                "need 1 <= fitsBelowPercent <= dumpAbovePercent < 100"
            }
        }
    }

    // Version 2 checks. A batch under minBatchSize is SMALL; a share within nearLinePoints of the dump line gets a second sample.
    data class Check(
        val minBatchSize: Int = 30,
        val nearLinePoints: Int = 5,
    ) {
        init {
            require(minBatchSize >= 1) { "minBatchSize must be at least 1" }
            require(nearLinePoints in 0..100) { "nearLinePoints must be 0..100" }
        }
    }
}

enum class RelevanceMode {
    OFF,
    RECORD_ONLY,
    AUTO_HIDE,
    // Old names, kept so a server with the old variable still boots: shadow runs as record-only, flag as
    // version 1's flag mode (RelevanceSwitch).
    SHADOW,
    FLAG,
}
