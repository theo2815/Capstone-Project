package com.quickpitik.service.disbursement

import org.junit.jupiter.api.Test
import java.math.BigDecimal
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// The HTTP call is owed (KYB-gated, verify-in-test-mode). What IS testable is
// the pure request-building: money conversion, the reference key, and the
// institution mapping.
class PayMongoDisbursementProviderTest {

    private val req = DisbursementRequest(
        cycleId = "PAY-JUANDC-007",
        amountPhp = BigDecimal("1234.50"),
        method = "gcash",
        accountNumber = "09171234567",
        accountName = "Juan Dela Cruz",
    )

    @Test
    fun `converts pesos to centavos exactly`() {
        assertEquals(123450L, PayMongoDisbursementProvider.toCentavos(BigDecimal("1234.50")))
        assertEquals(50000L, PayMongoDisbursementProvider.toCentavos(BigDecimal("500")))
    }

    @Test
    fun `builds a transfer carrying centavos and the cycle id as reference`() {
        val attrs = PayMongoDisbursementProvider.buildTransferRequest(req).data.attributes
        assertEquals(123450L, attrs.amount)
        assertEquals("PHP", attrs.currency)
        assertEquals("PAY-JUANDC-007", attrs.referenceNumber)
        assertEquals("09171234567", attrs.destination.accountNumber)
        assertEquals("Juan Dela Cruz", attrs.destination.accountHolderName)
    }

    @Test
    fun `maps each e-wallet to an institution code`() {
        assertEquals("GXCHPHM2XXX", PayMongoDisbursementProvider.bankCodeFor("gcash"))
        assertEquals("PAPHPHM1XXX", PayMongoDisbursementProvider.bankCodeFor("maya"))
        assertEquals("GOTYPHM1XXX", PayMongoDisbursementProvider.bankCodeFor("gotyme"))
    }

    @Test
    fun `rejects an unknown method`() {
        assertFailsWith<IllegalArgumentException> {
            PayMongoDisbursementProvider.bankCodeFor("paypal")
        }
    }
}
