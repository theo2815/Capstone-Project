package com.quickpitik.controller

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.orders.PaymongoTransferWebhookEvent
import com.quickpitik.exception.ValidationException
import com.quickpitik.security.PaymongoSignatureVerifier
import com.quickpitik.service.disbursement.PayoutDisbursementWebhookService
import jakarta.servlet.http.HttpServletRequest
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

// Inbound PayMongo TRANSFER (disbursement) webhook — separate from the payment
// webhook so payout confirmations and buyer-payment confirmations stay in their
// own handlers. Same path prefix (/payments/webhook/**), so WebhookRawBodyFilter
// caches the raw body for the HMAC verifier and SecurityConfig already permits
// it unauthenticated — authorization is the signature. Only meaningful once the
// PayMongo disbursement adapter is enabled (Phase 11); harmless otherwise.
@RestController
@RequestMapping("/api/v1/payments/webhook")
class PaymongoTransferWebhookController(
    private val verifier: PaymongoSignatureVerifier,
    private val service: PayoutDisbursementWebhookService,
    private val objectMapper: ObjectMapper,
) {
    @PostMapping("/paymongo-transfers")
    fun handle(
        request: HttpServletRequest,
        @RequestBody(required = false) rawBody: String?,
    ): Map<String, Any?> {
        verifier.verify(request)
        val body = try {
            objectMapper.readValue(rawBody ?: "", PaymongoTransferWebhookEvent::class.java)
        } catch (_: JsonProcessingException) {
            throw ValidationException("Invalid webhook payload.")
        }
        return service.handle(body)
    }
}
