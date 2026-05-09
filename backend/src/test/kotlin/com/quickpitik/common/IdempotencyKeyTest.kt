package com.quickpitik.common

import com.quickpitik.exception.ValidationException
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNull

class IdempotencyKeyTest {

    @Test
    fun `null and blank parse to null`() {
        assertNull(IdempotencyKey.parse(null))
        assertNull(IdempotencyKey.parse(""))
        assertNull(IdempotencyKey.parse("   "))
    }

    @Test
    fun `valid UUID round-trips lower-cased`() {
        val raw = UUID.randomUUID().toString().uppercase()
        val parsed = IdempotencyKey.parse(raw)!!
        assertEquals(raw.lowercase(), parsed.value)
    }

    @Test
    fun `non-UUID input throws ValidationException with INVALID_IDEMPOTENCY_KEY`() {
        val ex = assertThrows<ValidationException> { IdempotencyKey.parse("not-a-uuid") }
        assertEquals(ErrorCodes.INVALID_IDEMPOTENCY_KEY, ex.code)
    }
}
