package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

@ConfigurationProperties(prefix = "app.ai-api")
data class AiApiProperties(
    // Master switch — when false, all server-side ai-api calls are skipped.
    // Photo upload still succeeds (faces + bibs are best-effort already), runner
    // selfie upload skips the quality gate, runner face-search short-circuits to
    // 503 AI_API_UNAVAILABLE. Flip to true via AI_API_ENABLED=true when ai-api
    // is running and you want face/bib indexing + search to work.
    val enabled: Boolean = true,
    val baseUrl: String = "http://localhost:8000",
    val apiKey: String = "dev-only-ai-api-key-DO-NOT-USE-IN-PRODUCTION",
    val connectTimeout: Duration = Duration.ofSeconds(5),
    val readTimeout: Duration = Duration.ofSeconds(30),
    val maxRetries: Int = 3,
    val backoffBaseMillis: Long = 500L,
    val faceMatchThresholdDefault: Double = 0.6,
    val faceTopKDefault: Int = 5,
    val bibConfidenceThresholdDefault: Double = 0.7,
)
