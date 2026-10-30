package com.quickpitik.common

import com.quickpitik.config.PaymongoProperties
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class ProductionSecretsGuardTest {

    private fun guard(
        jwt: String = "real-jwt-secret-that-is-long-enough-for-hs256-signing-ok",
        capability: String = "real-capability-secret",
        seed: String = "real-seed-secret",
        webhook: String = "real-webhook-secret",
        paymongoKey: String = "sk_live_real",
        paymongoWebhook: String = "whsk_real",
        resend: String = "re_real",
        adminPassword: String = "a-strong-operator-password",
        storage: String = "S3",
        originSecret: String = "real-origin-secret",
        paymongo: PaymongoProperties = PaymongoProperties(),
    ) = ProductionSecretsGuard(
        jwt, capability, seed, webhook, paymongoKey, paymongoWebhook, resend, adminPassword, storage, originSecret,
        paymongo,
    )

    @Test
    fun `boots when every secret is real and storage is S3`() {
        guard()
    }

    @Test
    fun `names every dev placeholder still in place`() {
        val ex = assertThrows(IllegalStateException::class.java) {
            guard(
                jwt = "dev-only-secret-DO-NOT-USE-IN-PRODUCTION",
                resend = "re_dev-only-DO-NOT-USE-IN-PRODUCTION",
                adminPassword = "changeme123",
                storage = "LOCAL",
            )
        }
        val expected = listOf("JWT_SECRET", "RESEND_API_KEY", "ADMIN_BOOTSTRAP_PASSWORD", "STORAGE_BACKEND")
        assertEquals(expected, expected.filter { it in ex.message.orEmpty() })
        assertEquals(false, "PAYMONGO_SECRET_KEY" in ex.message.orEmpty())
    }

    @Test
    fun `rejects missing and dev origin secrets`() {
        for (originSecret in listOf("", "dev-only-origin-secret")) {
            val ex = assertThrows(IllegalStateException::class.java) { guard(originSecret = originSecret) }
            assertEquals(true, "ORIGIN_SECRET" in ex.message.orEmpty())
        }
    }

    @Test
    fun `boots with several https PayMongo refund link prefixes`() {
        guard(paymongo = PaymongoProperties(refundLinkAllowedPrefixes = listOf("https://transfer.paymongo.com/", "https://pay.paymongo.com/x/")))
    }

    @Test
    fun `rejects refund link prefixes that are not https PayMongo addresses`() {
        val bad = listOf(
            listOf("https://example.test/"),
            listOf("http://transfer.paymongo.com/"),
            listOf("https://transfer.paymongo.com"),
            listOf("https://paymongo.com.evil.tld/"),
            listOf("https://user@transfer.paymongo.com/"),
            emptyList(),
            listOf("https://transfer.paymongo.com/", "https://example.test/"),
        )
        for (prefixes in bad) {
            val ex = assertThrows(IllegalStateException::class.java, { guard(paymongo = PaymongoProperties(refundLinkAllowedPrefixes = prefixes)) }, "$prefixes")
            assertEquals(true, "PAYMONGO_REFUND_LINK_ALLOWED_PREFIXES" in ex.message.orEmpty(), "$prefixes")
        }
    }
}
