package com.quickpitik.service.disbursement

import com.quickpitik.config.PlatformProperties
import org.junit.jupiter.api.Test
import java.math.BigDecimal
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class DisbursementServiceTest {

    private val req = DisbursementRequest(
        cycleId = "PAY-X-001",
        amountPhp = BigDecimal("500"),
        method = "gcash",
        accountNumber = "09171234567",
        accountName = "Juan Dela Cruz",
    )

    @Test
    fun `resolves the active provider by name and delegates`() {
        val svc = DisbursementService(listOf(ManualDisbursementProvider()), PlatformProperties())
        assertEquals("manual", svc.activeName)
        // Manual disbursement is a no-op — no external transfer id.
        assertNull(svc.initiate(req).providerTransferId)
    }

    @Test
    fun `throws when the configured provider is not registered`() {
        // e.g. disbursement-provider=paymongo but the adapter bean isn't present.
        val props = PlatformProperties(disbursementProvider = "paymongo")
        val svc = DisbursementService(listOf(ManualDisbursementProvider()), props)
        assertFailsWith<IllegalStateException> { svc.initiate(req) }
    }
}
