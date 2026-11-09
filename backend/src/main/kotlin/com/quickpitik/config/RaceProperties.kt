package com.quickpitik.config

import com.quickpitik.entity.RaceType
import org.springframework.boot.context.properties.ConfigurationProperties

// AA-R1: shared race events (auto-approval Plan 2). Shares are 0-1.
@ConfigurationProperties(prefix = "app.races")
data class RaceProperties(
    // Rollout switch: photographer race create, join, edit and race price.
    val selfServeEnabled: Boolean = false,
    val dateWindowPastDays: Long = 3,
    val dateWindowAheadDays: Long = 90,
    // Per Manila day, per photographer.
    val dailyCreateCap: Int = 3,
    val duplicate: Duplicate = Duplicate(),
    val proof: Proof = Proof(),
    val sameBib: SameBib = SameBib(),
    val reports: Reports = Reports(),
) {
    data class Duplicate(
        val strongScore: Double = 0.85,
        val mediumScore: Double = 0.5,
        val dateToleranceDays: Long = 1,
    )

    data class Proof(
        val minPhotographers: Int = 2,
        val minIndexedPhotos: Int = 20,
        val minBibShare: Double = 0.30,
        // Below this indexed share the proof sweep only flags.
        val minIndexedShare: Double = 0.5,
        val sweepIntervalMs: Long = 3_600_000,
        // AA-R8: for these types a failed bib rule only flags (plates are rarely read).
        val bibFlagOnlyTypes: Set<RaceType> = setOf(RaceType.CYCLING),
    )

    data class SameBib(val minSharedBibs: Int = 30)

    // Distinct reporting accounts that hide an unproven race.
    data class Reports(val hideThreshold: Int = 3)
}
