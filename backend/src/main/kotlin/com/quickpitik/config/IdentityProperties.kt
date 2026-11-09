package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

// Auto-approval Plan 1: photographer identity (AWS Face Liveness + CompareFaces).
// Scores use AWS's 0-100 scale, unlike RekognitionProperties, because admins see 0-100.
@ConfigurationProperties(prefix = "app.identity")
data class IdentityProperties(
    // Off = today's upload rule; on = uploads need AUTO_APPROVED or APPROVED (AA-I6).
    val gateEnabled: Boolean = false,
    // Face Liveness only (AA-0.1); every other Rekognition call stays in app.ai.rekognition.region.
    val livenessRegion: String = "ap-southeast-5",
    // Strict until measured (review F6). [NOT MEASURED: set from AA-0.4]
    val livenessMinConfidence: Double = 95.0,
    val faceMinSimilarity: Double = 99.0,
    // Sessions per photographer per Manila day.
    val livenessAttemptsPerDay: Int = 3,
    // Role the backend assumes for the browser's keys (AA-0.2). Blank = 503 LIVENESS_UNAVAILABLE.
    val livenessRoleArn: String = "",
    val livenessCredentialSeconds: Int = 900,
) {
    init {
        require(livenessMinConfidence in 0.0..100.0) { "app.identity.liveness-min-confidence must be within 0..100" }
        require(faceMinSimilarity in 0.0..100.0) { "app.identity.face-min-similarity must be within 0..100" }
        require(livenessAttemptsPerDay >= 1) { "app.identity.liveness-attempts-per-day must be at least 1" }
        // STS minimum 900 s; the role's MaxSessionDuration is the default 3600 s.
        require(livenessCredentialSeconds in 900..3600) { "app.identity.liveness-credential-seconds must be within 900..3600" }
    }
}
