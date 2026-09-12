package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.remote.PhotographerMessageDto
import com.quickpitik.mobile.data.remote.VerificationSubmitResponseDto
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

// The setup checklist is keyed off the backend's stable missingCodes: a step is
// done when its code is NOT in the missing set. Locks that mapping (and the
// SETUP_CODES_ALL skew-fallback) so a wording/enum drift can't silently
// mis-check a step — the substring-matching bug this logic replaced. The
// verificationSummary cases below lock the five derivations the Overview reads
// off the same response.
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

    // --- verificationSummary: the one read of the verification state ---

    private fun verification(status: String, codes: List<String>? = null) =
        VerificationSubmitResponseDto(status, null, codes, null, null)

    private fun message(kind: String, body: String) =
        PhotographerMessageDto("m1", kind, null, body, null, "2026-09-12T00:00:00Z", null)

    @Test
    fun `a missing response reads as incomplete, and only approved is live`() {
        val loading = verificationSummary(null, null)
        assertEquals("incomplete", loading.status)
        assertFalse(loading.isApproved)
        assertFalse(loading.isRejected)
        // A still-loading state must not mark the checklist done OR undone.
        assertEquals(emptyList<String>(), loading.missingCodes)

        assertTrue(verificationSummary(verification("APPROVED"), null).isApproved)
        assertFalse(verificationSummary(verification("pending"), null).isApproved)
    }

    @Test
    fun `a rejection needs the status AND the inbox message that carries the reason`() {
        val reason = message("verification_rejected", "Your cover photo is too small.")
        val rejected = verificationSummary(verification("rejected"), reason)
        assertTrue(rejected.isRejected)
        assertEquals("Your cover photo is too small.", rejected.rejectionReason)
        // The arm that makes isRejected worth having: the backend reports
        // "incomplete" and only the message says it was a rejection. Every
        // consumer reads `isRejected || status == "rejected"`, so losing this
        // drops the banner and the reason on the common path.
        val incomplete = verificationSummary(verification("incomplete"), reason)
        assertTrue(incomplete.isRejected)
        assertEquals("Your cover photo is too small.", incomplete.rejectionReason)
        // Same message, but the studio is live again — not a rejection any more.
        assertFalse(verificationSummary(verification("approved"), reason).isRejected)
        // Status says rejected, but no message says why: no reason to show.
        val silent = verificationSummary(verification("rejected"), message("payout_paid", "Payout sent."))
        assertFalse(silent.isRejected)
        assertNull(silent.rejectionReason)
    }

    @Test
    fun `a non-complete status with no codes falls back to all of them, never to none`() {
        // Version skew: null codes on an incomplete profile must not read as
        // "every step done" (100% for a blank profile).
        assertEquals(SETUP_CODES_ALL, verificationSummary(verification("incomplete"), null).missingCodes)
        assertEquals(SETUP_CODES_ALL, verificationSummary(verification("rejected"), null).missingCodes)
        // approved and pending legitimately omit codes.
        assertEquals(emptyList<String>(), verificationSummary(verification("approved"), null).missingCodes)
        assertEquals(emptyList<String>(), verificationSummary(verification("pending"), null).missingCodes)
        // An empty list is an answer, not an omission: nothing is missing.
        assertEquals(emptyList<String>(), verificationSummary(verification("incomplete", emptyList()), null).missingCodes)
        assertEquals(listOf("REGION"), verificationSummary(verification("incomplete", listOf("REGION")), null).missingCodes)
    }
}
