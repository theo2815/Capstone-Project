package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.remote.SellerDetailsDto
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ShouldResumeUploadsTest {
    @Test fun `resumes on auto_approved`() {
        assertTrue(shouldResumeUploads(SellerDetailsDto("approved", null, "auto_approved", true)))
    }

    @Test fun `uploadsAllowed false never resumes`() {
        listOf("submitted", "approved", "missing", "rejected").forEach { status ->
            assertFalse(status, shouldResumeUploads(SellerDetailsDto(status, null, uploadsAllowed = false)))
        }
    }

    @Test fun `null uploadsAllowed falls back to the soft gate`() {
        listOf("submitted", "approved").forEach { status ->
            assertTrue(status, shouldResumeUploads(SellerDetailsDto(status, null)))
        }
        listOf("missing", "rejected", "something_new").forEach { status ->
            assertFalse(status, shouldResumeUploads(SellerDetailsDto(status, null)))
        }
        assertFalse(shouldResumeUploads(null))
    }

    @Test fun `an explicit upload answer wins over the legacy status`() {
        assertTrue(shouldResumeUploads(SellerDetailsDto("submitted", null, "manual_review", true)))
    }
}
