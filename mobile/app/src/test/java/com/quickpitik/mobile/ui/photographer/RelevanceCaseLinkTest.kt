package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.remote.PhotographerMessageDto
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
}
