package com.quickpitik.mobile.ui.photographer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

// The setup checklist is keyed off the backend's stable missingCodes: a step is
// done when its code is NOT in the missing set. Locks that mapping (and the
// SETUP_CODES_ALL skew-fallback) so a wording/enum drift can't silently
// mis-check a step — the substring-matching bug this logic replaced.
class SetupItemsTest {

    @Test
    fun `no missing codes means every step is complete`() {
        val steps = buildSetupItems(emptyList())
        assertEquals(8, steps.size)
        assertTrue(steps.all { it.isCompleted })
    }

    @Test
    fun `all requirement codes missing means no step is complete`() {
        // Also proves SETUP_CODES_ALL (the skew fallback) covers every step —
        // any step whose code is absent from it would read as done here.
        assertEquals(8, SETUP_CODES_ALL.size)
        val steps = buildSetupItems(SETUP_CODES_ALL)
        assertTrue(steps.none { it.isCompleted })
    }

    @Test
    fun `a single missing code marks only that step incomplete`() {
        val steps = buildSetupItems(listOf("PROFILE_PHOTO"))
        assertFalse(steps.first { it.title == "Avatar" }.isCompleted)
        assertEquals(7, steps.count { it.isCompleted })
    }
}
