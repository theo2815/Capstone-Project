package com.quickpitik.service.storage

import org.junit.jupiter.api.Test
import org.springframework.http.MediaType
import kotlin.test.assertEquals

// The selfie keys handed to ai-api get their content type from here; a wrong
// answer is a rejected face search, so pin every arm.
class ContentTypesTest {

    @Test
    fun `a key maps to the content type its extension implies`() {
        assertEquals(MediaType.IMAGE_JPEG_VALUE, contentTypeForKey("selfies/u/a.jpg"))
        assertEquals(MediaType.IMAGE_JPEG_VALUE, contentTypeForKey("selfies/u/a.JPEG"))
        assertEquals(MediaType.IMAGE_PNG_VALUE, contentTypeForKey("a.png"))
        assertEquals("image/webp", contentTypeForKey("a.webp"))
        assertEquals(MediaType.APPLICATION_OCTET_STREAM_VALUE, contentTypeForKey("a.heic"))
        assertEquals(MediaType.APPLICATION_OCTET_STREAM_VALUE, contentTypeForKey("no-extension"))
    }
}
