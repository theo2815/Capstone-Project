package com.quickpitik.common

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
    ) = ProductionSecretsGuard(
        jwt, capability, seed, webhook, paymongoKey, paymongoWebhook, resend, adminPassword, storage,
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
}
