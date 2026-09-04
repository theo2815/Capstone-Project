package com.quickpitik.controller

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.orders.PaymongoWebhookEvent
import com.quickpitik.security.PaymongoSignatureVerifier
import com.quickpitik.service.orders.PaymongoWebhookService
import jakarta.servlet.http.HttpServletRequest
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

// PayMongo webhook endpoint. The literal /webhook/paymongo path wins
// over PaymentWebhookController's /webhook/{provider} wildcard at the
// Spring routing layer (more-specific matches first), so PayMongo
// deliveries flow here while ops/stub deliveries keep flowing to the
// generic handler.
//
// Auth: public path (SecurityConfig permits POST /api/v1/payments/webhook/**).
// Authorization is enforced entirely by the HMAC verifier below; an
// unsigned or wrongly-signed delivery raises a 401 and never reaches
// the service layer. WebhookRawBodyFilter wraps every request under
// /api/v1/payments/webhook/** so the verifier can read the raw bytes
// post-Jackson.
@RestController
@RequestMapping("/api/v1/payments/webhook")
class PaymongoWebhookController(
    private val verifier: PaymongoSignatureVerifier,
    private val service: PaymongoWebhookService,
    private val objectMapper: ObjectMapper,
) {
    @PostMapping("/paymongo")
    fun handle(
        request: HttpServletRequest,
        @RequestBody rawBody: String,
    ): Map<String, Any?> {
        verifier.verify(request)
        return service.handle(objectMapper.readValue(rawBody, PaymongoWebhookEvent::class.java))
    }
}
