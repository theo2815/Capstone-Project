package com.quickpitik.controller

import com.quickpitik.dto.orders.PaymentWebhookRequest
import com.quickpitik.dto.orders.PaymentWebhookResponse
import com.quickpitik.service.orders.PaymentWebhookService
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
) {
    // Provider-stub. Real HMAC-signed webhooks will land in Phase H polish; for v1
    // this is a public endpoint so ops can curl-drive a PENDING order to PAID for
    // a demo. SecurityConfig allow-lists POST /payments/webhook/**.
    @PostMapping("/webhook/{provider}")
    fun handle(
        @PathVariable provider: String,
        @Valid @RequestBody body: PaymentWebhookRequest,
    ): PaymentWebhookResponse =
        webhookService.handle(provider = provider, request = body)
}
