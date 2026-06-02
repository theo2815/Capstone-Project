package com.quickpitik.dto.ai

import com.fasterxml.jackson.annotation.JsonIgnoreProperties

// Inbound ai-api job webhook body. ai-api sends only the job id + a count/error —
// the actual results are fetched via GET /jobs/{id} during ingest. Unknown fields
// are ignored so ai-api can add payload keys without breaking the receiver.
@JsonIgnoreProperties(ignoreUnknown = true)
data class AiWebhookEvent(
    val event: String = "",
    val job_id: String = "",
    val result_count: Int? = null,
    val error: String? = null,
)
