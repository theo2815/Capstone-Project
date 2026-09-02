package com.quickpitik.service.image

import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.GradientPaint
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import javax.imageio.IIOImage
import javax.imageio.ImageIO
import javax.imageio.ImageWriteParam
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// The fingerprint behind POST /public/photos/verify. What matters: the same
// picture re-saved, shrunk, or screenshotted still lands within the match
// threshold, and a different picture does not.
class PerceptualHashTest {

    @Test
    fun `hash is deterministic for the same pixels`() {
        assertEquals(PerceptualHash.of(scene()), PerceptualHash.of(scene()))
    }

    @Test
    fun `survives a 50 percent downscale plus JPEG recompression`() {
        val original = scene()
        val degraded = jpegRoundTrip(scale(original, 0.5), quality = 0.6f)

        val distance = PerceptualHash.distance(PerceptualHash.of(original), PerceptualHash.of(degraded))

        assertTrue(distance <= 8, "expected <= 8 bits of drift, got $distance")
    }

    @Test
    fun `a different picture is far away`() {
        val distance = PerceptualHash.distance(PerceptualHash.of(scene()), PerceptualHash.of(otherScene()))

        assertTrue(distance >= 16, "expected >= 16 bits apart, got $distance")
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    // Never a flat image: a horizontal gradient with a dark runner-shaped blob
    // left of center and a bright "bib" rectangle, so the low frequencies carry
    // real structure for the DCT to lock onto.
    private fun scene(): BufferedImage = canvas { g ->
        g.paint = GradientPaint(0f, 0f, Color(40, 60, 90), 256f, 0f, Color(220, 210, 190))
        g.fillRect(0, 0, 256, 256)
        g.color = Color(20, 20, 25)
        g.fillOval(60, 40, 70, 170)
        g.color = Color(245, 245, 240)
        g.fillRect(80, 120, 40, 28)
    }

    // Vertical gradient, blob on the other side, big dark ground band.
    private fun otherScene(): BufferedImage = canvas { g ->
        g.paint = GradientPaint(0f, 0f, Color(230, 235, 240), 0f, 256f, Color(90, 70, 50))
        g.fillRect(0, 0, 256, 256)
        g.color = Color(30, 30, 30)
        g.fillRect(0, 190, 256, 66)
        g.color = Color(200, 40, 40)
        g.fillOval(160, 30, 80, 80)
    }

    private fun canvas(draw: (java.awt.Graphics2D) -> Unit): BufferedImage {
        val img = BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB)
        val g = img.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            draw(g)
        } finally {
            g.dispose()
        }
        return img
    }

    private fun scale(src: BufferedImage, factor: Double): BufferedImage {
        val out = BufferedImage((src.width * factor).toInt(), (src.height * factor).toInt(), BufferedImage.TYPE_INT_RGB)
        val g = out.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            g.drawImage(src, 0, 0, out.width, out.height, null)
        } finally {
            g.dispose()
        }
        return out
    }

    private fun jpegRoundTrip(src: BufferedImage, quality: Float): BufferedImage {
        val writer = ImageIO.getImageWritersByFormatName("jpg").next()
        val params = writer.defaultWriteParam.apply {
            compressionMode = ImageWriteParam.MODE_EXPLICIT
            compressionQuality = quality
        }
        val bytes = ByteArrayOutputStream()
        ImageIO.createImageOutputStream(bytes).use { ios ->
            writer.output = ios
            writer.write(null, IIOImage(src, null, null), params)
        }
        writer.dispose()
        return ImageIO.read(ByteArrayInputStream(bytes.toByteArray()))
    }
}
