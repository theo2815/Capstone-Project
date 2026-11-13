package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

// AA-I11 profile-image moderation (DetectModerationLabels). A label is blocked
// when its name or its parent's name is listed, so one list covers taxonomy v6 and v7.
@ConfigurationProperties(prefix = "app.moderation")
data class ModerationProperties(
    // 0-1 like the other thresholds; sent to AWS as ×100. Settled by AA-0.5 (2026-11-08): 0.8 holds on race photos; the true-positive side is not tested.
    val imageMinConfidence: Double = 0.8,
    val imageBlockedLabels: List<String> = listOf(
        "Explicit Nudity", "Explicit", "Non-Explicit Nudity of Intimate parts and Kissing",
        "Violence", "Visually Disturbing", "Hate Symbols",
    ),
) {
    init {
        require(imageMinConfidence in 0.0..1.0) { "app.moderation.image-min-confidence must be within 0..1" }
    }
}
