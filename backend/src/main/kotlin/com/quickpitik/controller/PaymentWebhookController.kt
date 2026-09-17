package com.quickpitik.controller

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.orders.PaymentWebhookRequest
import com.quickpitik.dto.orders.PaymentWebhookResponse
import com.quickpitik.security.WebhookSignatureVerifier
import com.quickpitik.service.orders.PaymentWebhookService
import jakarta.servlet.http.HttpServletRequest
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/v1/payments")
class PaymentWebhookController(
    private val webhookService: PaymentWebhookService,
    private val signatureVerifier: WebhookSignatureVerifier,
    private val objectMapper: ObjectMapper,
) {
    // Public endpoint (whitelisted in SecurityConfig) — providers hit us
    // server-to-server with no Bearer JWT. Authorization comes entirely from
    // the X-QuickPitik-Signature HMAC over the raw JSON body.
    // WebhookRawBodyFilter wraps the request so signatureVerifier can read the
    // raw bytes; the body stays a String until the signature is verified, so an
    // unsigned delivery is a 401 whatever it contains (blank providerRef is
    // refused by the service).
    @PostMapping("/webhook/{provider}")
    fun handle(
        request: HttpServletRequest,
        @PathVariable provider: String,
        @RequestBody rawBody: String,
    ): PaymentWebhookResponse {
        signatureVerifier.verify(request)
        val body = objectMapper.readValue(rawBody, PaymentWebhookRequest::class.java)
        return webhookService.handle(provider = provider, request = body)
    }
}
