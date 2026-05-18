package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

// PayMongo Checkout Sessions wiring. Real values are env-driven and never
// committed — dev defaults are intentionally fake so an accidental boot
// without `application-local.yml` doesn't quietly hit PayMongo with garbage.
//
// `secretKey`        — sk_test_… for HTTP Basic auth (PayMongo uses the
//                      secret key as the username, empty password).
// `webhookSecret`    — whsk_… signing secret for the `Paymongo-Signature`
//                      header on inbound webhooks. Format is
//                      `t=<unix>,te=<test_sig>,li=<live_sig>` — verifier
//                      lives in Phase 3.
// `successUrl`       — where PayMongo redirects the user after pay
//                      (Phase 2 appends `?orderId=…&token=…`).
// `cancelUrl`        — where PayMongo redirects after user cancels.
// `baseUrl`          — PayMongo REST API root.
@ConfigurationProperties(prefix = "app.payments.paymongo")
data class PaymongoProperties(
    val secretKey: String = "sk_test_dev-only-DO-NOT-USE-IN-PRODUCTION",
    val webhookSecret: String = "whsk_dev-only-DO-NOT-USE-IN-PRODUCTION",
    val successUrl: String = "http://localhost:3000/orders/return",
    val cancelUrl: String = "http://localhost:3000/cart",
    val baseUrl: String = "https://api.paymongo.com/v1",
    val connectTimeout: Duration = Duration.ofSeconds(5),
    val readTimeout: Duration = Duration.ofSeconds(15),
)
