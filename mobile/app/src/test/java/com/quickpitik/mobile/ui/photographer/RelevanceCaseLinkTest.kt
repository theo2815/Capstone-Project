package com.quickpitik.mobile.ui.photographer

import com.google.gson.Gson
import com.quickpitik.mobile.data.remote.PhotographerMessageDto
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.WarningOrange
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

// RV-3.6: a relevance note carries the case id; the inbox row links to the
// website's case page built from that id alone.
class RelevanceCaseLinkTest {

    private fun note(kind: String, caseId: String?) = PhotographerMessageDto(
        "m1", kind, "t", "b", null, "2026-11-16T00:00:00Z", null, caseId,
    )

    @Test
    fun `the url is built from the id only`() {
        val id = "3f2b9c1e-0000-4000-8000-000000000001"
        assertEquals("https://quickpitik.com/dashboard/relevance/$id", relevanceCaseUrl(id))
        // A hostile id stays one path segment on the same host.
        assertEquals(
            "https://quickpitik.com/dashboard/relevance/a%2Fb%3Fc%20d",
            relevanceCaseUrl("a/b?c d"),
        )
    }

    @Test
    fun `a note with a case id shows the button and one without does not`() {
        assertEquals(
            "https://quickpitik.com/dashboard/relevance/abc",
            relevanceCaseLinkUrl(note("relevance_notice", "abc")),
        )
        assertNull(relevanceCaseLinkUrl(note("relevance_notice", null)))
        assertNull(relevanceCaseLinkUrl(note("relevance_notice", " ")))
    }

    @Test
    fun `the relevance kind reads Uploads in the warning tone`() {
        assertEquals("Uploads", messageKindLabel("relevance_notice"))
        assertEquals(WarningOrange, messageToneColor("relevance_notice"))
    }

    @Test
    fun `an unknown kind keeps its fallback label and tone`() {
        assertEquals("Some new kind", messageKindLabel("some_new_kind"))
        assertEquals(Slate, messageToneColor("some_new_kind"))
    }

    @Test
    fun `every older kind keeps its label and tone`() {
        val expected = mapOf(
            "verification_approved" to ("Verification approved" to Fresh),
            "verification_rejected" to ("Verification needs changes" to WarningOrange),
            "verification_reset" to ("Verification reset" to WarningOrange),
            "suspended" to ("Account suspended" to ErrorRed),
            "unsuspended" to ("Account reinstated" to Fresh),
            "force_edit" to ("Profile updated by admin" to Slate),
            "dispute_resolved" to ("Refund resolved" to Fresh),
            "dispute_denied" to ("Refund denied" to WarningOrange),
            "dispute_escalated" to ("Refund escalated" to WarningOrange),
            "payout_approved" to ("Payout approved" to Fresh),
            "payout_held" to ("Payout held" to WarningOrange),
            "payout_paid" to ("Payout paid" to Fresh),
            "payout_report_acknowledged" to ("Report acknowledged" to Ink),
            "payout_report_resolved" to ("Report resolved" to Fresh),
            "admin_message" to ("Message from admin" to Ink),
        )
        expected.forEach { (kind, labelAndTone) ->
            assertEquals(kind, labelAndTone.first, messageKindLabel(kind))
            assertEquals(kind, labelAndTone.second, messageToneColor(kind))
        }
    }

    // The app parses with plain Gson (RetrofitClient), which maps by field name.
    @Test
    fun `the case id is read from the backend json name and is null without it`() {
        val base = """"id":"m1","kind":"relevance_notice","title":"t","body":"b","createdAt":"2026-11-16T00:00:00Z""""
        val gson = Gson()
        val with = gson.fromJson("""{$base,"relevanceCaseId":"3f2b9c1e"}""", PhotographerMessageDto::class.java)
        val without = gson.fromJson("{$base}", PhotographerMessageDto::class.java)
        assertEquals("3f2b9c1e", with.relevanceCaseId)
        assertNull(without.relevanceCaseId)
    }
}
