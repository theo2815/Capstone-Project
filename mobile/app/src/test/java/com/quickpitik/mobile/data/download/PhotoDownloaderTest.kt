package com.quickpitik.mobile.data.download

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** The two pure helpers behind every purchased-photo save: the file name and the server-sent name. */
class PhotoDownloaderTest {
    @Test fun `filenames keep the bib readable and take eight id characters to stay unique`() {
        assertEquals("quickpitik-bib-b-4082-1a2b3c4d.jpg", PhotoDownloader.buildFilename("1a2b3c4d-ffff", " B-4082 "))
        assertEquals("quickpitik-untagged-1a2b3c4d.jpg", PhotoDownloader.buildFilename("1a2b3c4d-ffff", "   "))
        assertEquals("quickpitik-untagged-1a2b3c4d.jpg", PhotoDownloader.buildFilename("1a2b3c4d-ffff", null))
    }

    @Test fun `content-disposition prefers the RFC 5987 name over the quoted one`() {
        assertEquals(
            "cebu run.zip",
            PhotoDownloader.parseContentDispositionFilename("attachment; filename=\"fallback.zip\"; filename*=UTF-8''cebu%20run.zip"),
        )
        assertEquals("photos.zip", PhotoDownloader.parseContentDispositionFilename("attachment; filename=\"photos.zip\""))
        assertEquals("photo.jpg", PhotoDownloader.parseContentDispositionFilename("attachment; filename=photo.jpg"))
        assertNull(PhotoDownloader.parseContentDispositionFilename("inline"))
        assertNull(PhotoDownloader.parseContentDispositionFilename(null))
    }
}
