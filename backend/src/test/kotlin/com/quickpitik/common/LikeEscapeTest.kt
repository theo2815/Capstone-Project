package com.quickpitik.common

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class LikeEscapeTest {
    @Test
    fun `escapeLike makes percent, underscore and backslash plain text`() {
        assertEquals("50\\%", escapeLike("50%"))
        assertEquals("a\\_b", escapeLike("a_b"))
        assertEquals("a\\\\b", escapeLike("a\\b"))
        assertEquals("", escapeLike(""))
        assertEquals("cebu marathon", escapeLike("cebu marathon"))
    }
}
