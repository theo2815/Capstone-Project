package com.quickpitik.mobile.ui.photographer

import org.junit.Assert.assertEquals
import org.junit.Test

// The sync card shows the latest failed upload's stored text (backend message or
// network exception). Locks the rewrite into plain sentences, and that anything
// unrecognised passes through unchanged.
class FriendlyUploadErrorTest {

    @Test
    fun `duplicate in another event names that event`() {
        assertEquals(
            "This photo is already in your event \"TEST MOBILE\", so it wasn't added here. Nothing to fix.",
            friendlyUploadError("This photo already exists in your event 'TEST MOBILE'."),
        )
    }

    @Test
    fun `duplicate without an event name says another event`() {
        assertEquals(
            "This photo is already in another of your events, so it wasn't added here. Nothing to fix.",
            friendlyUploadError("This photo already exists in your event 'another event'."),
        )
    }

    @Test
    fun `event and file rejections become plain sentences`() {
        assertEquals("This event isn't accepting uploads right now. Check its dates or status.",
            friendlyUploadError("This event is not accepting uploads."))
        assertEquals("This event belongs to another photographer, so photos can't be added to it.",
            friendlyUploadError("This event belongs to another photographer."))
        listOf(
            "Upload must be image/jpeg, image/png, or image/webp.",
            "Upload must be image/jpeg, image/png, or image/webp under 25 MB.",
            "Image dimensions exceed the supported maximum.",
        ).forEach {
            assertEquals("This file can't be uploaded. Use a JPEG, PNG or WebP under 25 MB.", friendlyUploadError(it))
        }
        assertEquals("The upload didn't finish. Tap Retry to send it again.",
            friendlyUploadError("The uploaded file wasn't found in storage. Upload it again."))
        assertEquals("The photo didn't reach cloud storage. Tap Retry to send it again.",
            friendlyUploadError("Storage upload returned HTTP 403."))
        assertEquals("QuickPitik didn't accept this photo. Tap Retry. If it keeps happening, contact support.",
            friendlyUploadError("Upload rejected by server."))
    }

    @Test
    fun `network failures say check your connection`() {
        listOf(
            "Upload interrupted.",
            "timeout",
            "Unable to resolve host \"api.quickpitik.com\": No address associated with hostname",
            "Failed to connect to api.quickpitik.com/104.18.0.1:443",
            "Software caused connection abort",
        ).forEach {
            assertEquals("Couldn't reach QuickPitik. Check your connection. Photos retry on their own.",
                friendlyUploadError(it))
        }
    }

    @Test
    fun `clear or unknown messages pass through unchanged`() {
        val watermark = "Your watermark image is missing from storage. Re-upload it in Settings before uploading photos."
        assertEquals(watermark, friendlyUploadError(watermark))
        assertEquals("Sign in to resume uploads.", friendlyUploadError("Sign in to resume uploads."))
        assertEquals("Something new", friendlyUploadError("Something new"))
    }
}
