package com.quickpitik.dto.ai

import com.fasterxml.jackson.annotation.JsonIgnoreProperties

@JsonIgnoreProperties(ignoreUnknown = true)
data class AiApiEnvelope<T>(
    val success: Boolean = false,
    val data: T? = null,
    val error: AiApiError? = null,
    val request_id: String? = null,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class AiApiError(
    val code: String? = null,
    val message: String? = null,
)
