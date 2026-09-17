package com.quickpitik.controller

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.ai.AiWebhookEvent
import com.quickpitik.security.AiWebhookSignatureVerifier
import com.quickpitik.service.ai.AiWebhookService
import jakarta.servlet.http.HttpServletRequest
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

// Inbound ai-api job webhook (Phase C, prod path). Public route — authorization
// is the HMAC verifier; WebhookRawBodyFilter caches the raw bytes so the
// signature can be checked, and the body is parsed only after it verifies.
// Ingest runs off-thread; this handler just acks. Best-effort delivery → the
// poll backstop ingests anything missed, and ingest is idempotent so a
// duplicate webhook is a no-op.
@RestController
@RequestMapping("/api/v1/internal/ai-webhooks")
class AiWebhookController(
    private val verifier: AiWebhookSignatureVerifier,
    private val service: AiWebhookService,
    private val objectMapper: ObjectMapper,
) {
    @PostMapping
    fun handle(
        request: HttpServletRequest,
        @RequestBody rawBody: String,
    ): Map<String, Any?> {
        verifier.verify(request)
        val body = objectMapper.readValue(rawBody, AiWebhookEvent::class.java)
        if (body.job_id.isNotBlank() && (body.event == "job.completed" || body.event == "job.failed")) {
            service.onJobEvent(body.job_id)
        }
        return mapOf("received" to true)
    }
}
