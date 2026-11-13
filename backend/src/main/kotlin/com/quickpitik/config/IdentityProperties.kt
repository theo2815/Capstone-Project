package com.quickpitik.config

import com.quickpitik.entity.SellerIdType
import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.LocalDate

// Auto-approval Plan 1: photographer identity (AWS Face Liveness + CompareFaces).
// Scores use AWS's 0-100 scale, unlike RekognitionProperties, because admins see 0-100.
@ConfigurationProperties(prefix = "app.identity")
data class IdentityProperties(
    // Off = today's upload rule; on = uploads need AUTO_APPROVED or APPROVED (AA-I6).
    val gateEnabled: Boolean = false,
    // The announced turn-on date, shown to photographers (C3); blank = none. An invalid date fails boot.
    val gateStartsOn: LocalDate? = null,
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
    // Keys the stored ID-number hash (AA-I4). Never rotate it casually: older rows stop matching (AA-I5).
    val idNumberHmacSecret: String = "dev-only-id-number-hmac-secret",
    // A regex over the normalised number, per ID type. Blank = an admin checks this ID type. [NOT MEASURED: set from AA-0.5]
    val idNumberFormats: Map<SellerIdType, String> = emptyMap(),
    // The identity notice (AA-I22) sends one mail at a time, at most this many a second. [NOT MEASURED: Resend's limit]
    val noticeSendPerSecond: Int = 2,
) {
    /** Compiled once, so an invalid regex fails boot. */
    val idNumberPatterns: Map<SellerIdType, Regex> = idNumberFormats.filterValues { it.isNotBlank() }.mapValues { (type, regex) ->
        runCatching { Regex(regex) }.getOrElse { throw IllegalArgumentException("app.identity.id-number-formats.$type is not a valid regex") }
    }

    init {
        require(idNumberHmacSecret.isNotBlank()) { "app.identity.id-number-hmac-secret must not be blank" }
        require(livenessMinConfidence in 0.0..100.0) { "app.identity.liveness-min-confidence must be within 0..100" }
        require(faceMinSimilarity in 0.0..100.0) { "app.identity.face-min-similarity must be within 0..100" }
        require(livenessAttemptsPerDay >= 1) { "app.identity.liveness-attempts-per-day must be at least 1" }
        // STS minimum 900 s; the role's MaxSessionDuration is the default 3600 s.
        require(livenessCredentialSeconds in 900..3600) { "app.identity.liveness-credential-seconds must be within 900..3600" }
        require(noticeSendPerSecond in 1..1000) { "app.identity.notice-send-per-second must be within 1..1000" }
    }
}
