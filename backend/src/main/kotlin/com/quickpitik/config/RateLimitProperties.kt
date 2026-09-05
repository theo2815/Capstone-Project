package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

@ConfigurationProperties(prefix = "app.rate-limit")
data class RateLimitProperties(
    val enabled: Boolean = true,
    val photographerUpload: Policy = Policy(capacity = 600, refillPeriod = Duration.ofMinutes(1)),
    val publicGallery: Policy = Policy(capacity = 60, refillPeriod = Duration.ofMinutes(1)),
    // NFR-S-11: face/bib photo search 30 req / 15 min. Each face search burns
    // AI inference, so it costs far more than an ordinary read. bucket4j's
    // greedy refill = burst of 30, then ~1 token / 30 s.
    val photoSearch: Policy = Policy(capacity = 30, refillPeriod = Duration.ofMinutes(15)),
    // NFR-S-11: auth endpoints 10 req / 15 min by source IP — burst of 10,
    // then ~1 token / 90 s. Anything past that is credential stuffing, email
    // bombing, or token brute-force, not a human retrying.
    val authLogin: Policy = Policy(capacity = 10, refillPeriod = Duration.ofMinutes(15)),
    val authRegister: Policy = Policy(capacity = 10, refillPeriod = Duration.ofMinutes(15)),
    val authForgotPassword: Policy = Policy(capacity = 10, refillPeriod = Duration.ofMinutes(15)),
    // Its own bucket (not reset-password's): OTP guessing must not drain the
    // budget of the confirm step, and vice versa.
    val authVerifyResetOtp: Policy = Policy(capacity = 10, refillPeriod = Duration.ofMinutes(15)),
    val authResetPassword: Policy = Policy(capacity = 10, refillPeriod = Duration.ofMinutes(15)),
    // Guest checkout mints a PayMongo session per call — bound the burn rate
    // per IP.
    val orderCreate: Policy = Policy(capacity = 10, refillPeriod = Duration.ofMinutes(1)),
    // Return pages poll every two seconds for one minute; leave room for the
    // final detail request and a second tab without leaving reads unlimited.
    val orderRead: Policy = Policy(capacity = 90, refillPeriod = Duration.ofMinutes(1)),
    // `?verify=true` status checks retrieve the Payment Intent from PayMongo
    // per hit. The checkout sends at most ~3/min per order; leave headroom for
    // a second tab.
    val orderVerify: Policy = Policy(capacity = 10, refillPeriod = Duration.ofMinutes(1)),
    // Public coupon preview: one lookup + a page of photos per call. Its own
    // bucket so typo'd "Apply" clicks can't drain the checkout bucket.
    val couponPreview: Policy = Policy(capacity = 30, refillPeriod = Duration.ofMinutes(1)),
    // Public token-gated bundle download streams one S3 GET per photo + zip
    // CPU per call.
    val bundleDownload: Policy = Policy(capacity = 6, refillPeriod = Duration.ofMinutes(1)),
    // Public per-photo download (bundle route with `?photo=`, free-event
    // originals): one storage GET streamed through, no zip. Sized so a runner
    // saving a whole order one photo at a time stays under it.
    val photoDownload: Policy = Policy(capacity = 30, refillPeriod = Duration.ofMinutes(1)),
    // Authenticated small-file uploads (avatar / selfie / photographer cover /
    // watermark / payout QR) share one per-user bucket — selfie uploads can
    // trigger AI quality inference, the rest are storage writes.
    val mediaUpload: Policy = Policy(capacity = 20, refillPeriod = Duration.ofMinutes(1)),
    // Public photo verification: a decode + pHash + full-table Hamming scan per
    // call, reachable by anyone. Same posture as face search.
    val photoVerify: Policy = Policy(capacity = 10, refillPeriod = Duration.ofMinutes(15)),
) {
    data class Policy(
        val capacity: Long,
        val refillPeriod: Duration,
    )
}
