package com.quickpitik.service.image

import org.junit.jupiter.api.Test
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// Header-only decompression-bomb gate. The budget is parameterized so the
// "over budget" case uses a real 100×100 PNG against a tiny cap instead of a
// multi-hundred-MB fixture.
class ImagePixelGuardTest {

    private fun png(width: Int, height: Int): ByteArray {
        val out = ByteArrayOutputStream()
        ImageIO.write(BufferedImage(width, height, BufferedImage.TYPE_INT_RGB), "png", out)
        return out.toByteArray()
    }

    @Test
    fun `an image within budget passes`() {
        assertFalse(ImagePixelGuard.exceedsPixelBudget(png(100, 100)))
    }

    @Test
    fun `declared dimensions over budget are rejected without a full decode`() {
        assertTrue(ImagePixelGuard.exceedsPixelBudget(png(100, 100), maxPixels = 9_999))
    }

    @Test
    fun `dimensions exactly at budget pass`() {
        assertFalse(ImagePixelGuard.exceedsPixelBudget(png(100, 100), maxPixels = 10_000))
    }

    @Test
    fun `unreadable bytes are not this gate's problem`() {
        // No registered reader claims garbage — the later full decode produces
        // the existing unreadable-image 415, so the guard stays silent.
        assertFalse(ImagePixelGuard.exceedsPixelBudget(byteArrayOf(1, 2, 3, 4)))
    }
}
