package com.quickpitik.service.image

import org.junit.jupiter.api.Test
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.util.Base64
import javax.imageio.ImageIO
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// Header-only decompression-bomb gate. The over-budget case uses a
// header-only production-cap fixture without allocating the declared raster.
class ImagePixelGuardTest {

    companion object {
        // 10,000 × 8,001 = 80,010,000 pixels, above MAX_PIXELS, without
        // allocating the declared raster.
        fun overBudgetPngHeader(): ByteArray = byteArrayOf(
            -119, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
            0, 0, 39, 16, 0, 0, 31, 65, 8, 6, 0, 0, 0, -26, 16, 82, 64,
        )
    }

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
        assertTrue(ImagePixelGuard.exceedsPixelBudget(overBudgetPngHeader()))
    }

    @Test
    fun `WebP header dimensions over budget are rejected before raster decode`() {
        val bytes = Base64.getDecoder().decode(
            "UklGRjIAAABXRUJQVlA4ICYAAACQAQCdASoQABAAAUAmJYgCdLoAA5gA/vPfZrQtCBz+N5vNasQAAA==",
        )
        assertFalse(ImagePixelGuard.exceedsPixelBudget(bytes))
        // VP8 frame header: 10,000 x 8,001, without allocating that raster.
        bytes[26] = 0x10
        bytes[27] = 0x27
        bytes[28] = 0x41
        bytes[29] = 0x1f

        assertTrue(ImagePixelGuard.exceedsPixelBudget(bytes))
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
