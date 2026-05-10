package com.quickpitik.controller

import com.quickpitik.dto.orders.PaymentWebhookRequest
import com.quickpitik.dto.orders.PaymentWebhookResponse
import com.quickpitik.security.WebhookSignatureVerifier
import com.quickpitik.service.orders.PaymentWebhookService
import jakarta.servlet.http.HttpServletRequest
import jakarta.validation.Valid
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
) {
    // Public endpoint (whitelisted in SecurityConfig) — providers hit us
    // server-to-server with no Bearer JWT. Authorization comes entirely from
    // the X-QuickPitik-Signature HMAC over the raw JSON body.
    // WebhookRawBodyFilter wraps the request so signatureVerifier can read the
    // pre-deserialization bytes; verification runs before any business logic
    // touches the order.
    @PostMapping("/webhook/{provider}")
    fun handle(
        request: HttpServletRequest,
        @PathVariable provider: String,
        @Valid @RequestBody body: PaymentWebhookRequest,
    ): PaymentWebhookResponse {
        signatureVerifier.verify(request)
        return webhookService.handle(provider = provider, request = body)
    }
}
