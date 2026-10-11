package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

@ConfigurationProperties(prefix = "app.photo-removal")
data class PhotoRemovalProperties(
    val photographerResponseWindow: Duration = Duration.ofHours(72),
    val cleanupInterval: Duration = Duration.ofMinutes(1),
    val cleanupBackoff: Duration = Duration.ofMinutes(5),
    val sweepBatchSize: Int = 50,
    val inRequestCleanupJobs: Int = 5,
) {
    init {
        require(photographerResponseWindow > Duration.ZERO)
        require(cleanupInterval > Duration.ZERO)
        require(cleanupBackoff > Duration.ZERO)
        require(sweepBatchSize in 1..100)
        require(inRequestCleanupJobs in 0..100)
    }
}
