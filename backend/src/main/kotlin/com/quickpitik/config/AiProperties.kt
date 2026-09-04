package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

// Provider-level AI config, above the ai-api-specific settings in
// AiApiProperties. Selects which FaceBibProvider fulfils face + bib inference.
// The AI master switch (`enabled`) still lives on app.ai-api.
@ConfigurationProperties(prefix = "app.ai")
data class AiProperties(
    val provider: AiProvider = AiProvider.AI_API,
)

enum class AiProvider {
    AI_API,
    REKOGNITION,
}
