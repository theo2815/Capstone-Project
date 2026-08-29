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
// `successUrl`       — where PayMongo redirects the user after pay.
// `cancelUrl`        — where PayMongo redirects after user cancels.
// `baseUrl`          — PayMongo REST API root.
@ConfigurationProperties(prefix = "app.payments.paymongo")
data class PaymongoProperties(
    val secretKey: String = "sk_test_dev-only-DO-NOT-USE-IN-PRODUCTION",
    val webhookSecret: String = "whsk_dev-only-DO-NOT-USE-IN-PRODUCTION",
    val successUrl: String = "http://localhost:3000/orders/return",
    val cancelUrl: String = "http://localhost:3000/cart",
    // Mobile redirect targets. PayMongo redirects the in-browser tab to these
    // after pay/cancel for orders created with clientPlatform="android". The
    // bridge endpoints (MobileReturnController) emit HTML that opens
    // `quickpitik://` deep links, returning the user to the app. Default
    // points at the Android-emulator-reachable backend address; override per
    // environment via PAYMONGO_MOBILE_SUCCESS_URL / PAYMONGO_MOBILE_CANCEL_URL.
    val mobileSuccessUrl: String = "http://10.0.2.2:8080/api/v1/orders/mobile-return",
    val mobileCancelUrl: String = "http://10.0.2.2:8080/api/v1/orders/mobile-cancel",
    val baseUrl: String = "https://api.paymongo.com/v1",
    val connectTimeout: Duration = Duration.ofSeconds(5),
    val readTimeout: Duration = Duration.ofSeconds(15),
    val checkoutTtl: Duration = Duration.ofMinutes(30),
)
