package com.quickpitik.common

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.math.BigDecimal
import kotlin.test.assertEquals

class MoneyTest {

    @Test
    fun `pesos become centavos`() {
        assertEquals(12500L, BigDecimal("125.00").toCentavos())
        assertEquals(123456L, BigDecimal("1234.56").toCentavos())
        assertEquals(50000L, BigDecimal("500").toCentavos())
        assertEquals(0L, BigDecimal.ZERO.toCentavos())
    }

    @Test
    fun `a fraction of a centavo throws instead of truncating`() {
        assertThrows<ArithmeticException> { BigDecimal("0.005").toCentavos() }
    }
}
