package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

// Resend transactional email wiring. Real API key is env-driven; the
// `re_dev-only` default is intentionally invalid so a misconfigured boot
// fails loudly at the first send instead of pretending to dispatch.
//
// `apiKey`        — re_… bearer token for the Resend REST API.
// `fromAddress`   — sender address. Free tier delivers from
//                   `onboarding@resend.dev` only to the verified signup
//                   address. Swap to `receipts@quickpitik.com` (or similar)
//                   once the production sender domain is DNS-verified.
// `fromName`      — display name in the From header.
// `baseUrl`       — Resend REST API root.
@ConfigurationProperties(prefix = "app.email.resend")
data class ResendProperties(
    val apiKey: String = "re_dev-only-DO-NOT-USE-IN-PRODUCTION",
    val fromAddress: String = "onboarding@resend.dev",
    val fromName: String = "QuickPitik",
    val baseUrl: String = "https://api.resend.com",
    val connectTimeout: Duration = Duration.ofSeconds(5),
    val readTimeout: Duration = Duration.ofSeconds(15),
)
