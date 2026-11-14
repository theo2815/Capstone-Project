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
    val words: Words = Words(),
    val llm: Llm = Llm(),
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
        // Photos picked for one sample round; a smaller batch is sampled whole.
        val sampleSize: Int = 300,
        // A batch still uploading is checked again once it grew by this many photos.
        val countTrigger: Int = 500,
        // Check tick cadence, and the first delay after boot.
        val tickIntervalMs: Long = 15_000,
        val maxBatchesPerTick: Int = 3,
        // How long a claim on a batch holds before another server copy may take it.
        val claimMinutes: Long = 10,
        // Sampled photos measured per tick (one DetectLabels call each), on `workers` threads of their own.
        val measuresPerTick: Int = 60,
        val workers: Int = 2,
        // A photo that hits a throttle, 5xx or timeout this many times is given up (GAVE_UP), never junk.
        val measureMaxTries: Int = 5,
        // A case waits this long for version 1 to score its upload, then is decided with the copy and date results as they stand.
        val waitForRulesMinutes: Long = 30,
        // Model calls per case; when the last one fails the case is stored as can't tell (CALL_FAILED).
        val decideMaxAttempts: Int = 3,
    ) {
        init {
            require(minBatchSize >= 1) { "minBatchSize must be at least 1" }
            require(nearLinePoints in 0..100) { "nearLinePoints must be 0..100" }
            require(sampleSize >= 1) { "sampleSize must be at least 1" }
            require(countTrigger >= 1) { "countTrigger must be at least 1" }
            require(tickIntervalMs >= 1) { "tickIntervalMs must be at least 1" }
            require(maxBatchesPerTick >= 1) { "maxBatchesPerTick must be at least 1" }
            require(claimMinutes >= 1) { "claimMinutes must be at least 1" }
            require(measuresPerTick >= 1) { "measuresPerTick must be at least 1" }
            require(workers >= 1) { "workers must be at least 1" }
            require(measureMaxTries >= 1) { "measureMaxTries must be at least 1" }
            require(waitForRulesMinutes >= 0) { "waitForRulesMinutes must not be negative" }
            require(decideMaxAttempts >= 1) { "decideMaxAttempts must be at least 1" }
        }
    }

    // Version 2 words read off the photos (G14): once a batch settles, words on under this share of its counted
    // photos are deleted. topN: at most this many of the batch's words go into the summary sent to the model (G13).
    data class Words(
        val minSharePercent: Int = 3,
        val topN: Int = 30,
    ) {
        init {
            require(minSharePercent in 1..100) { "minSharePercent must be 1..100" }
            require(topN in 1..100) { "topN must be 1..100" }
        }
    }

    // Version 2 verdict: the model behind the check (O6, Anthropic direct). A blank apiKey means no call is ever made.
    // maxAttempts counts the first call; retryBackoff is the wait before each retry of a 429, 5xx or timeout.
    data class Llm(
        val apiKey: String = "",
        val baseUrl: String = "https://api.anthropic.com",
        val model: String = "claude-sonnet-5-5",
        val connectTimeout: Duration = Duration.ofSeconds(5),
        val readTimeout: Duration = Duration.ofSeconds(30),
        val maxAttempts: Int = 3,
        val retryBackoff: Duration = Duration.ofSeconds(1),
        // Room for the model's thinking as well as the short JSON answer; an answer cut off here is can't tell.
        val maxTokens: Int = 4096,
    ) {
        init {
            require(maxAttempts in 1..5) { "maxAttempts must be 1..5" }
            require(maxTokens >= 1) { "maxTokens must be at least 1" }
            require(connectTimeout >= Duration.ofMillis(1) && readTimeout >= Duration.ofMillis(1)) { "timeouts must be at least 1ms" }
            require(!retryBackoff.isNegative) { "retryBackoff must not be negative" }
        }

        // The key is a secret: keep it out of any log line that prints the settings.
        override fun toString() = "Llm(apiKey=${if (apiKey.isBlank()) "<unset>" else "<set>"}, baseUrl=$baseUrl, model=$model, " +
            "connectTimeout=$connectTimeout, readTimeout=$readTimeout, maxAttempts=$maxAttempts, retryBackoff=$retryBackoff, " +
            "maxTokens=$maxTokens)"
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
