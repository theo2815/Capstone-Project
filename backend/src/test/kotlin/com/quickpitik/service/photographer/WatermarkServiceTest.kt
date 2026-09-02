package com.quickpitik.service.photographer

import com.drew.imaging.ImageMetadataReader
import com.drew.metadata.xmp.XmpDirectory
import com.quickpitik.service.image.PerceptualHash
import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.GradientPaint
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// The preview every runner sees. Contract: JPEG ≤1280 on the long edge, the
// QuickPitik credit layer reaches every region of the frame (so no crop
// removes it), the credit is also carried as XMP metadata, and the pHash of
// the marked image stays close enough to the unmarked one that either variant
// verifies.
class WatermarkServiceTest {

    private val service = WatermarkService()
    private val photoId = UUID.fromString("7b3a9c1e-2d4f-4a6b-8c0d-1e2f3a4b5c6d")
    private val credit = WatermarkCredit(name = "Ana Reyes Studio", handle = "anareyes", photoId = photoId)

    @Test
    fun `output is a JPEG capped at 1280 on the long edge`() {
        val marked = service.processThumbnail(photoJpeg(2000, 1500), logoPng(), credit)

        val decoded = ImageIO.read(ByteArrayInputStream(marked.jpeg))
        assertEquals(1280, decoded.width)
        assertEquals(960, decoded.height)
    }

    @Test
    fun `credit layer touches every region of the frame`() {
        val marked = ImageIO.read(ByteArrayInputStream(service.processThumbnail(photoJpeg(2000, 1500), logoPng(), credit).jpeg))
        val plain = scaleToLongEdge(ImageIO.read(ByteArrayInputStream(photoJpeg(2000, 1500))), 1280)

        // 4×4 grid; each cell must carry some of the mark. This is the
        // crop-proof property — a runner cropping any quarter still sees it.
        val cellW = marked.width / 4
        val cellH = marked.height / 4
        for (cy in 0 until 4) for (cx in 0 until 4) {
            var changed = 0
            for (y in cy * cellH until (cy + 1) * cellH) for (x in cx * cellW until (cx + 1) * cellW) {
                if (channelDelta(marked.getRGB(x, y), plain.getRGB(x, y)) >= 12) changed++
            }
            val ratio = changed.toDouble() / (cellW * cellH)
            assertTrue(ratio > 0.01, "cell ($cx,$cy) untouched by the mark: ${"%.3f".format(ratio)} changed")
        }
    }

    @Test
    fun `credit is embedded as XMP metadata`() {
        val marked = service.processThumbnail(photoJpeg(1200, 800), logoPng(), credit)

        val xmp = ImageMetadataReader.readMetadata(ByteArrayInputStream(marked.jpeg))
            .getFirstDirectoryOfType(XmpDirectory::class.java)
        val props = xmp?.xmpProperties ?: emptyMap()
        assertEquals("Ana Reyes Studio", props["dc:creator[1]"], "dc:creator missing in $props")
        assertEquals(photoId.toString(), props["quickpitik:photoId"], "photoId missing in $props")
        assertEquals("QuickPitik", props["photoshop:Credit"])
    }

    @Test
    fun `same photo and credit render identical bytes`() {
        val a = service.processThumbnail(photoJpeg(1200, 800), logoPng(), credit)
        val b = service.processThumbnail(photoJpeg(1200, 800), logoPng(), credit)

        assertContentEquals(a.jpeg, b.jpeg)
        assertEquals(a.phash, b.phash)
    }

    @Test
    fun `a different photo id shifts the tile pattern`() {
        val a = ImageIO.read(ByteArrayInputStream(service.processThumbnail(photoJpeg(1200, 800), logoPng(), credit).jpeg))
        val b = ImageIO.read(
            ByteArrayInputStream(
                service.processThumbnail(photoJpeg(1200, 800), logoPng(), credit.copy(photoId = UUID.randomUUID())).jpeg,
            ),
        )

        val samePixels = (0 until a.width step 7).sumOf { x -> (0 until a.height step 7).count { y -> a.getRGB(x, y) == b.getRGB(x, y) } }
        val total = (a.width / 7 + 1) * (a.height / 7 + 1)
        assertFalse(samePixels > total * 0.9, "tile pattern did not move with the photo id")
    }

    @Test
    fun `phash of the marked preview stays within the verify threshold of the unmarked one`() {
        val marked = service.processThumbnail(photoJpeg(2000, 1500), logoPng(), credit)
        val plain = scaleToLongEdge(ImageIO.read(ByteArrayInputStream(photoJpeg(2000, 1500))), 1280)

        val distance = PerceptualHash.distance(marked.phash, PerceptualHash.of(plain))
        assertTrue(distance <= 12, "marked vs unmarked drifted $distance bits")
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun photoJpeg(w: Int, h: Int): ByteArray {
        val img = BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
        val g = img.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            // Sky-to-road gradient with a dark runner blob and a bright bib —
            // both bright and dark regions, so adaptive polarity gets exercised.
            g.paint = GradientPaint(0f, 0f, Color(120, 170, 230), 0f, h.toFloat(), Color(70, 70, 75))
            g.fillRect(0, 0, w, h)
            g.color = Color(25, 25, 30)
            g.fillOval(w / 3, h / 5, w / 6, h * 3 / 5)
            g.color = Color(250, 250, 245)
            g.fillRect(w / 3 + w / 24, h / 2, w / 12, h / 12)
        } finally {
            g.dispose()
        }
        val out = ByteArrayOutputStream()
        ImageIO.write(img, "jpg", out)
        return out.toByteArray()
    }

    private fun logoPng(): ByteArray {
        val img = BufferedImage(300, 100, BufferedImage.TYPE_INT_ARGB)
        val g = img.createGraphics()
        try {
            g.color = Color(200, 30, 30)
            g.fillRoundRect(10, 10, 280, 80, 30, 30)
        } finally {
            g.dispose()
        }
        val out = ByteArrayOutputStream()
        ImageIO.write(img, "png", out)
        return out.toByteArray()
    }

    private fun scaleToLongEdge(src: BufferedImage, max: Int): BufferedImage {
        val scale = max.toDouble() / maxOf(src.width, src.height)
        val out = BufferedImage((src.width * scale).toInt(), (src.height * scale).toInt(), BufferedImage.TYPE_INT_RGB)
        val g = out.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
            g.drawImage(src, 0, 0, out.width, out.height, null)
        } finally {
            g.dispose()
        }
        return out
    }

    private fun channelDelta(a: Int, b: Int): Int = maxOf(
        abs(((a shr 16) and 0xFF) - ((b shr 16) and 0xFF)),
        abs(((a shr 8) and 0xFF) - ((b shr 8) and 0xFF)),
        abs((a and 0xFF) - (b and 0xFF)),
    )
}
