package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

@ConfigurationProperties(prefix = "app.webhooks.payment")
data class PaymentWebhookProperties(
    val hmacSecret: String,
    val signatureHeader: String = "X-QuickPitik-Signature",
)
