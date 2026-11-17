package com.quickpitik.service.auth

import org.junit.jupiter.api.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class LegalVersionsTest {

    @Test fun `1·10 is above 1·9`() {
        assertTrue(atLeast("1.10", "1.9"))
        assertFalse(atLeast("1.9", "1.10"))
        assertTrue(atLeast("2.0", "1.9"))
    }

    @Test fun `an equal version is enough`() {
        assertTrue(atLeast("1.7", "1.7"))
        assertTrue(atLeast("1.7.0", "1.7"), "a missing part counts as 0")
    }

    @Test fun `no accepted version is never enough`() {
        assertFalse(atLeast(null, "1.7"))
        assertFalse(atLeast("", "1.7"))
    }

    @Test fun `a version that is not numbers is never enough`() {
        assertFalse(atLeast("1.7-draft", "1.7"))
        assertFalse(atLeast("v1.8", "1.7"))
        assertFalse(atLeast("1..8", "1.7"))
    }

    // RV-L4 review: edge cases, pinned as the code behaves today.
    @Test fun `a trailing dot is not a version`() {
        assertFalse(atLeast("1.7.", "1.7"))
        assertFalse(atLeast("1.8", "1.7."), "a minimum that does not parse is never met")
    }

    @Test fun `spaces around a version are not trimmed`() {
        assertFalse(atLeast(" 1.7", "1.7"))
        assertFalse(atLeast("1.7 ", "1.7"))
    }

    @Test fun `leading zeros compare as numbers`() {
        assertTrue(atLeast("1.07", "1.7"))
        assertTrue(atLeast("1.7", "1.07"))
        assertFalse(atLeast("1.06", "1.7"))
    }

    @Test fun `versions of different lengths compare with zeros`() {
        assertTrue(atLeast("1.7", "1.7.0"))
        assertFalse(atLeast("1.7", "1.7.1"))
        assertTrue(atLeast("1.7.1", "1.7"))
    }
}
