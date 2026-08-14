package com.quickpitik.service.image

import org.junit.jupiter.api.Test
import java.awt.image.BufferedImage
import kotlin.test.assertEquals
import kotlin.test.assertSame

// Shared by the photographer photo path (WatermarkService) and the runner selfie
// path (SelfieService). The contract that matters to both: orientation 1 is a
// pass-through, and a 90° tag swaps the output dimensions.
class ExifOrientationTest {

    private fun image(w: Int, h: Int) = BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)

    @Test
    fun `bytes with no EXIF read as NORMAL`() {
        assertEquals(ExifOrientation.NORMAL, ExifOrientation.read("not an image".toByteArray()))
    }

    @Test
    fun `NORMAL returns the very same instance — no needless re-encode`() {
        val src = image(40, 20)
        assertSame(src, ExifOrientation.apply(src, ExifOrientation.NORMAL))
    }

    @Test
    fun `orientation 6 rotates 90 degrees and swaps dimensions`() {
        // The iPhone-portrait case: landscape pixels tagged "rotate me".
        val rotated = ExifOrientation.apply(image(40, 20), 6)
        assertEquals(20, rotated.width)
        assertEquals(40, rotated.height)
    }

    @Test
    fun `orientation 3 rotates 180 degrees and keeps dimensions`() {
        val rotated = ExifOrientation.apply(image(40, 20), 3)
        assertEquals(40, rotated.width)
        assertEquals(20, rotated.height)
    }

    @Test
    fun `an unknown orientation tag is a defensive pass-through`() {
        val src = image(40, 20)
        assertSame(src, ExifOrientation.apply(src, 99))
    }
}
