package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

@ConfigurationProperties(prefix = "app.ai-api")
data class AiApiProperties(
    val baseUrl: String = "http://localhost:8000",
    val apiKey: String = "dev-only-ai-api-key-DO-NOT-USE-IN-PRODUCTION",
    val connectTimeout: Duration = Duration.ofSeconds(5),
    val readTimeout: Duration = Duration.ofSeconds(30),
    val maxRetries: Int = 3,
    val backoffBaseMillis: Long = 500L,
    val faceMatchThresholdDefault: Double = 0.6,
    val faceTopKDefault: Int = 5,
    val bibConfidenceThresholdDefault: Double = 0.7,
    val blurRejectThreshold: Double = 100.0,
)
