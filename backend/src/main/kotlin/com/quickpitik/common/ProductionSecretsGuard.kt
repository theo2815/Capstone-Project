package com.quickpitik.common

import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.autoconfigure.flyway.FlywayConfigurationCustomizer
import org.springframework.context.annotation.Profile
import org.springframework.stereotype.Component

/**
 * Refuses to boot under the `prod` profile while any secret is still the
 * in-repo dev placeholder, or the origin secret is blank. Dev defaults let
 * the app run locally, but do not otherwise trip a production runtime error
 * (the JWT placeholder is long enough for jjwt,
 * Resend silently logs instead of sending, LOCAL storage just writes to the
 * container disk). Failing at bean construction means this runs before any
 * ApplicationRunner — no bootstrap admin gets minted with `changeme123`.
 *
 * Implements [FlywayConfigurationCustomizer] (as a no-op) purely so the
 * `flyway` bean depends on this one: the check then fires before the first
 * database connection, and a misconfigured deploy fails with this message
 * rather than a migration or connection error.
 */
@Component
@Profile("prod")
class ProductionSecretsGuard(
    @Value("\${jwt.secret}") jwtSecret: String,
    @Value("\${app.platform.order-capability-secret}") orderCapabilitySecret: String,
    @Value("\${app.watermark.seed-secret}") watermarkSeedSecret: String,
    @Value("\${app.webhooks.payment.hmac-secret}") paymentWebhookSecret: String,
    @Value("\${app.payments.paymongo.secret-key}") paymongoSecretKey: String,
    @Value("\${app.payments.paymongo.webhook-secret}") paymongoWebhookSecret: String,
    @Value("\${app.email.resend.api-key}") resendApiKey: String,
    @Value("\${app.admin.bootstrap-password:}") adminBootstrapPassword: String,
    @Value("\${app.storage.backend}") storageBackend: String,
    @Value("\${app.security.origin-secret}") originSecret: String,
) : FlywayConfigurationCustomizer {
    override fun customize(configuration: org.flywaydb.core.api.configuration.FluentConfiguration) = Unit

    init {
        val offenders = offenders(
            jwtSecret = jwtSecret,
            orderCapabilitySecret = orderCapabilitySecret,
            watermarkSeedSecret = watermarkSeedSecret,
            paymentWebhookSecret = paymentWebhookSecret,
            paymongoSecretKey = paymongoSecretKey,
            paymongoWebhookSecret = paymongoWebhookSecret,
            resendApiKey = resendApiKey,
            adminBootstrapPassword = adminBootstrapPassword,
            storageBackend = storageBackend,
            originSecret = originSecret,
        )
        check(offenders.isEmpty()) {
            "Refusing to start with profile 'prod': invalid values for " +
                offenders.joinToString() + ". Set them as environment variables."
        }
    }

    companion object {
        private const val PLACEHOLDER = "dev-only"

        /** Env-var names whose values are still dev defaults. Pure, for the unit test. */
        fun offenders(
            jwtSecret: String,
            orderCapabilitySecret: String,
            watermarkSeedSecret: String,
            paymentWebhookSecret: String,
            paymongoSecretKey: String,
            paymongoWebhookSecret: String,
            resendApiKey: String,
            adminBootstrapPassword: String,
            storageBackend: String,
            originSecret: String,
        ): List<String> = buildList {
            if (PLACEHOLDER in jwtSecret) add("JWT_SECRET")
            if (PLACEHOLDER in orderCapabilitySecret) add("ORDER_CAPABILITY_SECRET")
            if (PLACEHOLDER in watermarkSeedSecret) add("WATERMARK_SEED_SECRET")
            if (PLACEHOLDER in paymentWebhookSecret) add("PAYMENT_WEBHOOK_HMAC_SECRET")
            if (PLACEHOLDER in paymongoSecretKey) add("PAYMONGO_SECRET_KEY")
            if (PLACEHOLDER in paymongoWebhookSecret) add("PAYMONGO_WEBHOOK_SECRET")
            if (PLACEHOLDER in resendApiKey) add("RESEND_API_KEY")
            if (adminBootstrapPassword == "changeme123") add("ADMIN_BOOTSTRAP_PASSWORD")
            if (!storageBackend.equals("S3", ignoreCase = true)) add("STORAGE_BACKEND")
            if (originSecret.isBlank() || PLACEHOLDER in originSecret) add("ORIGIN_SECRET")
        }
    }
}
