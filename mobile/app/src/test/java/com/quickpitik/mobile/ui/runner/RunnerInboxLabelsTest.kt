package com.quickpitik.mobile.ui.runner

import com.google.gson.Gson
import com.quickpitik.mobile.data.remote.RunnerMessageDto
import com.quickpitik.mobile.ui.common.photoDeletionStatusUrl
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class RunnerInboxLabelsTest {
    private fun message(kind: String, requestId: String? = null, orderId: String? = null) = RunnerMessageDto(
        id = "m1", kind = kind, title = null, body = "b", orderId = orderId, sourceDecisionId = null,
        createdAt = "2026-11-09T00:00:00Z", readAt = null, photoRemovalRequestId = requestId,
    )

    @Test fun `photo_removal_update is labelled Photo deletion`() {
        assertEquals("Photo deletion", runnerMessageKindLabel("photo_removal_update"))
        assertEquals("Photo removed", runnerMessageKindLabel("photo_removed"))
    }

    @Test fun `an unknown kind still shows Update`() {
        assertEquals("Update", runnerMessageKindLabel("something_new"))
    }

    @Test fun `a message without photoRemovalRequestId parses with null`() {
        val parsed = Gson().fromJson(
            """{"id":"m1","kind":"admin_message","body":"b","createdAt":"t"}""", RunnerMessageDto::class.java,
        )
        assertNull(parsed.photoRemovalRequestId)
    }

    @Test fun `a photo_removal_update row opens its status page`() {
        assertEquals(photoDeletionStatusUrl("r-1"), runnerMessageWebUrl(message("photo_removal_update", requestId = "r-1")))
    }

    @Test fun `a photo_removed row opens the order, not the deletion page`() {
        assertNull(runnerMessageWebUrl(message("photo_removed", requestId = "someone-elses-case", orderId = "o-1")))
    }
}
