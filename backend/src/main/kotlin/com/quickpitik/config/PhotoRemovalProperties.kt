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
    val runner: Runner = Runner(),
) {
    init {
        require(photographerResponseWindow > Duration.ZERO)
        require(cleanupInterval > Duration.ZERO)
        require(cleanupBackoff > Duration.ZERO)
        require(sweepBatchSize in 1..100)
        require(inRequestCleanupJobs in 0..100)
    }

    // Runner deletion requests. Liveness values are on AWS's 0-100 scale, similarity on 0-1.
    data class Runner(
        val livenessPassMin: Double = 90.0,
        val livenessUncertainMin: Double = 50.0,
        val autoHideMinSimilarity: Double = 0.99,
        val autoHideMaxPerRequest: Int = 30,
        val photographerDailyAutoHides: Int = 50,
        val candidateLimit: Int = 50,
        val emailClaimStaleAfter: Duration = Duration.ofMinutes(10),
        // RD-2: 0 closes intake (kill switch); at most this many open requests per runner.
        val checksPerDay: Int = 3,
        val maxOpenRequests: Int = 3,
    ) {
        init {
            require(livenessUncertainMin >= 0.0 && livenessUncertainMin <= livenessPassMin && livenessPassMin <= 100.0)
            require(autoHideMinSimilarity in 0.0..1.0)
            require(autoHideMaxPerRequest >= 0)
            require(photographerDailyAutoHides >= 0)
            require(candidateLimit in 1..4096)
            require(emailClaimStaleAfter > Duration.ZERO)
            require(checksPerDay >= 0 && maxOpenRequests >= 1)
        }
    }
}
