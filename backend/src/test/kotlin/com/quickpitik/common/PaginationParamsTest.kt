package com.quickpitik.common

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class PaginationParamsTest {

    @Test
    fun `defaults apply when nothing is supplied`() {
        val p = PaginationParams.of(null, null)
        assertEquals(0, p.offset)
        assertEquals(PaginationParams.DEFAULT_LIMIT, p.limit)
    }

    @Test
    fun `negative offset is clamped to zero`() {
        val p = PaginationParams.of(-3, 10)
        assertEquals(0, p.offset)
        assertEquals(10, p.limit)
    }

    @Test
    fun `limit is bounded by max`() {
        val p = PaginationParams.of(0, 9999)
        assertEquals(PaginationParams.MAX_LIMIT, p.limit)
    }

    @Test
    fun `limit is bounded below by one`() {
        val p = PaginationParams.of(0, 0)
        assertEquals(1, p.limit)
    }

    @Test
    fun `paginated response uses the params`() {
        val p = PaginationParams.of(50, 25)
        val r = PaginatedResponse.of(items = listOf("a", "b"), total = 100L, params = p)
        assertEquals(50, r.offset)
        assertEquals(25, r.limit)
        assertEquals(100L, r.total)
        assertEquals(2, r.items.size)
    }
}
