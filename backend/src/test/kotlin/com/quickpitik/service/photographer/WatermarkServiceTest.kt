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

    private val service = WatermarkService("test-seed-secret")
    private val photoId = UUID.fromString("7b3a9c1e-2d4f-4a6b-8c0d-1e2f3a4b5c6d")
    private val credit = WatermarkCredit(name = "Ana Reyes Studio", handle = "anareyes", photoId = photoId)

    @Test
    fun `output is a JPEG capped at 1280 on the long edge`() {
        val marked = service.processThumbnail(photoJpeg(2000, 1500), logoPng(), credit, platformMark = true)

        val decoded = ImageIO.read(ByteArrayInputStream(marked.jpeg))
        assertEquals(1280, decoded.width)
        assertEquals(960, decoded.height)
    }

    @Test
    fun `credit layer touches every region of the frame`() {
        val marked = ImageIO.read(ByteArrayInputStream(service.processThumbnail(photoJpeg(2000, 1500), logoPng(), credit, platformMark = true).jpeg))
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
    fun `statement block claims the centre of the frame`() {
        val marked = ImageIO.read(ByteArrayInputStream(service.processThumbnail(photoJpeg(2000, 1500), logoPng(), credit, platformMark = true).jpeg))
        val plain = scaleToLongEdge(ImageIO.read(ByteArrayInputStream(photoJpeg(2000, 1500))), 1280)

        // The middle band (where the runner's torso and bib sit) must carry a
        // dense mark — cropping to the subject cannot yield a clean copy.
        fun changedRatio(x0: Int, x1: Int, y0: Int, y1: Int): Double {
            var changed = 0
            for (y in y0 until y1) for (x in x0 until x1) {
                if (channelDelta(marked.getRGB(x, y), plain.getRGB(x, y)) >= 40) changed++
            }
            return changed.toDouble() / ((x1 - x0) * (y1 - y0))
        }
        val centre = changedRatio(marked.width / 4, marked.width * 3 / 4, marked.height * 2 / 5, marked.height * 3 / 5)
        val topEdge = changedRatio(0, marked.width, 0, marked.height / 10)
        assertTrue(centre > 0.10, "centre band too lightly marked: ${"%.3f".format(centre)}")
        assertTrue(centre > topEdge * 2, "centre ($centre) should be marked far heavier than the edge ($topEdge)")
    }

    @Test
    fun `credit is embedded as XMP metadata`() {
        val marked = service.processThumbnail(photoJpeg(1200, 800), logoPng(), credit, platformMark = true)

        val xmp = ImageMetadataReader.readMetadata(ByteArrayInputStream(marked.jpeg))
            .getFirstDirectoryOfType(XmpDirectory::class.java)
        val props = xmp?.xmpProperties ?: emptyMap()
        assertEquals("Ana Reyes Studio", props["dc:creator[1]"], "dc:creator missing in $props")
        assertEquals(photoId.toString(), props["quickpitik:photoId"], "photoId missing in $props")
        assertEquals("QuickPitik", props["photoshop:Credit"])
        // Machine-readable rights: the IPTC/PLUS AI opt-out, usage terms, and
        // the "special instructions" field — what compliant tools read.
        assertEquals("http://ns.useplus.org/ldf/vocab/DMI-PROHIBITED-AIGENAI", props["plus:DataMining"], "DataMining missing in $props")
        assertTrue(props["xmpRights:UsageTerms[1]"]?.contains("Do not remove") == true, "UsageTerms missing in $props")
        assertTrue(props["photoshop:Instructions"]?.contains("watermark") == true, "Instructions missing in $props")
    }

    @Test
    fun `same photo and credit render identical bytes`() {
        val a = service.processThumbnail(photoJpeg(1200, 800), logoPng(), credit, platformMark = true)
        val b = service.processThumbnail(photoJpeg(1200, 800), logoPng(), credit, platformMark = true)

        assertContentEquals(a.jpeg, b.jpeg)
        assertEquals(a.phash, b.phash)
        assertEquals(a.phashClean, b.phashClean)
        assertEquals(a.phashCentre, b.phashCentre)
    }

    // The seed is HMAC(secret, photoId), not the public photo id: nobody who
    // only knows the id can regenerate the exact layer and subtract it.
    @Test
    fun `a different seed secret moves the tile pattern`() {
        val a = ImageIO.read(ByteArrayInputStream(service.processThumbnail(photoJpeg(1200, 800), logoPng(), credit, platformMark = true).jpeg))
        val b = ImageIO.read(ByteArrayInputStream(WatermarkService("another-secret").processThumbnail(photoJpeg(1200, 800), logoPng(), credit, platformMark = true).jpeg))

        val samePixels = (0 until a.width step 7).sumOf { x -> (0 until a.height step 7).count { y -> a.getRGB(x, y) == b.getRGB(x, y) } }
        val total = (a.width / 7 + 1) * (a.height / 7 + 1)
        assertFalse(samePixels > total * 0.9, "tile pattern did not move with the seed secret")
    }

    // The stripe layer is not a rigid lattice: a straight line drawn on the
    // layer comes out wavy — displaced by a smooth per-photo field — and
    // unbroken, so template matching fails while coverage holds.
    @Test
    fun `warp displaces a straight line smoothly without breaking it`() {
        val w = 800
        val h = 600
        val pad = 24
        val layer = BufferedImage(w + 2 * pad, h + 2 * pad, BufferedImage.TYPE_INT_ARGB_PRE)
        val g = layer.createGraphics()
        try {
            g.color = Color.WHITE
            g.fillRect(pad + 200, 0, 3, layer.height)
        } finally {
            g.dispose()
        }

        val warped = service.warp(layer, pad, kotlin.random.Random(7))

        assertEquals(w, warped.width)
        assertEquals(h, warped.height)
        val lefts = (0 until h).map { y ->
            (0 until w).firstOrNull { x -> (warped.getRGB(x, y) ushr 24) > 128 }
                ?: error("row $y lost the line entirely")
        }
        assertTrue(lefts.max() - lefts.min() >= 4, "line was not displaced: ${lefts.min()}..${lefts.max()}")
        for (y in 1 until h) assertTrue(abs(lefts[y] - lefts[y - 1]) <= 2, "row $y jumped ${lefts[y - 1]} → ${lefts[y]}")
    }

    // Fingerprints taken BEFORE the mark is drawn: a copy someone has cleaned
    // still hashes to what we registered.
    @Test
    fun `clean and centre hashes are taken before the mark is drawn`() {
        val marked = service.processThumbnail(photoJpeg(2000, 1500), logoPng(), credit, platformMark = true)
        val plain = scaleToLongEdge(ImageIO.read(ByteArrayInputStream(photoJpeg(2000, 1500))), 1280)

        assertEquals(PerceptualHash.of(plain), marked.phashClean)
        assertEquals(PerceptualHash.ofCentre(plain), marked.phashCentre)
    }

    @Test
    fun `a cleaned, downscaled, recompressed full-frame copy matches the clean hash`() {
        val marked = service.processThumbnail(photoJpeg(2000, 1500), logoPng(), credit, platformMark = true)
        val plain = scaleToLongEdge(ImageIO.read(ByteArrayInputStream(photoJpeg(2000, 1500))), 1280)

        val copy = recompress(scaleToLongEdge(plain, 896), 0.6f)
        val distance = PerceptualHash.distance(PerceptualHash.of(copy), marked.phashClean)
        assertTrue(distance <= 6, "cleaned full-frame copy drifted $distance bits from the clean hash")
    }

    @Test
    fun `a centre crop of a cleaned copy matches the centre hash`() {
        val marked = service.processThumbnail(photoJpeg(2000, 1500), logoPng(), credit, platformMark = true)
        val plain = scaleToLongEdge(ImageIO.read(ByteArrayInputStream(photoJpeg(2000, 1500))), 1280)

        // The theft shape after screenshot: crop to the runner (middle 60%),
        // save at 70%, JPEG q60.
        val crop = plain.getSubimage(plain.width / 5, plain.height / 5, plain.width * 3 / 5, plain.height * 3 / 5)
        val copy = recompress(scaleToLongEdge(crop, (crop.width * 0.7).toInt()), 0.6f)
        val distance = PerceptualHash.distance(PerceptualHash.of(copy), marked.phashCentre)
        assertTrue(distance <= 12, "centre-cropped copy drifted $distance bits from the centre hash")
    }

    @Test
    fun `a different photo id shifts the tile pattern`() {
        val a = ImageIO.read(ByteArrayInputStream(service.processThumbnail(photoJpeg(1200, 800), logoPng(), credit, platformMark = true).jpeg))
        val b = ImageIO.read(
            ByteArrayInputStream(
                service.processThumbnail(photoJpeg(1200, 800), logoPng(), credit.copy(photoId = UUID.randomUUID()), platformMark = true).jpeg,
            ),
        )

        val samePixels = (0 until a.width step 7).sumOf { x -> (0 until a.height step 7).count { y -> a.getRGB(x, y) == b.getRGB(x, y) } }
        val total = (a.width / 7 + 1) * (a.height / 7 + 1)
        assertFalse(samePixels > total * 0.9, "tile pattern did not move with the photo id")
    }

    @Test
    fun `phash of the marked preview stays within the verify threshold of the unmarked one`() {
        val marked = service.processThumbnail(photoJpeg(2000, 1500), logoPng(), credit, platformMark = true)
        val plain = scaleToLongEdge(ImageIO.read(ByteArrayInputStream(photoJpeg(2000, 1500))), 1280)

        val distance = PerceptualHash.distance(marked.phash, PerceptualHash.of(plain))
        assertTrue(distance <= 12, "marked vs unmarked drifted $distance bits")
    }

    // ─── Free events (V46): the platform mark is a per-event policy ───────
    // A FREE photographer-owned event ships its previews without the QuickPitik
    // layers; the photographer's own logo is optional. The preview must still
    // be the ≤1280 derivative with all three fingerprints, so nothing in the
    // serving path ever falls back to the clean original.

    @Test
    fun `no platform mark and no logo renders the plain frame with its fingerprints`() {
        val marked = service.processThumbnail(photoJpeg(2000, 1500), null, credit, platformMark = false)
        val decoded = ImageIO.read(ByteArrayInputStream(marked.jpeg))
        val plain = scaleToLongEdge(ImageIO.read(ByteArrayInputStream(photoJpeg(2000, 1500))), 1280)

        assertEquals(1280, decoded.width)
        // Nothing was drawn: the marked hash IS the clean hash.
        assertEquals(marked.phashClean, marked.phash)
        assertEquals(PerceptualHash.ofCentre(plain), marked.phashCentre)
        var changed = 0
        for (y in 0 until decoded.height) for (x in 0 until decoded.width) {
            if (channelDelta(decoded.getRGB(x, y), plain.getRGB(x, y)) >= 40) changed++
        }
        assertTrue(changed < decoded.width * decoded.height * 0.005, "frame should be unmarked, $changed px changed")
    }

    @Test
    fun `no platform mark with a logo touches only the corner`() {
        val marked = ImageIO.read(
            ByteArrayInputStream(service.processThumbnail(photoJpeg(2000, 1500), logoPng(), credit, platformMark = false).jpeg),
        )
        val plain = scaleToLongEdge(ImageIO.read(ByteArrayInputStream(photoJpeg(2000, 1500))), 1280)

        fun changedRatio(x0: Int, x1: Int, y0: Int, y1: Int): Double {
            var changed = 0
            for (y in y0 until y1) for (x in x0 until x1) {
                if (channelDelta(marked.getRGB(x, y), plain.getRGB(x, y)) >= 40) changed++
            }
            return changed.toDouble() / ((x1 - x0) * (y1 - y0))
        }
        val centre = changedRatio(marked.width / 4, marked.width * 3 / 4, marked.height * 2 / 5, marked.height * 3 / 5)
        val corner = changedRatio(marked.width * 4 / 5, marked.width, marked.height * 4 / 5, marked.height)
        assertTrue(centre < 0.01, "centre band should be untouched without the platform mark: ${"%.3f".format(centre)}")
        assertTrue(corner > 0.05, "photographer logo missing from the corner: ${"%.3f".format(corner)}")
    }

    @Test
    fun `credit metadata is still embedded without the platform mark`() {
        val marked = service.processThumbnail(photoJpeg(1200, 800), null, credit, platformMark = false)

        val xmp = ImageMetadataReader.readMetadata(ByteArrayInputStream(marked.jpeg))
            .getFirstDirectoryOfType(XmpDirectory::class.java)
        assertEquals("Ana Reyes Studio", xmp?.xmpProperties?.get("dc:creator[1]"))
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

    private fun recompress(img: BufferedImage, quality: Float): BufferedImage {
        val writer = ImageIO.getImageWritersByFormatName("jpg").next()
        val params = writer.defaultWriteParam.apply {
            compressionMode = javax.imageio.ImageWriteParam.MODE_EXPLICIT
            compressionQuality = quality
        }
        val out = ByteArrayOutputStream()
        ImageIO.createImageOutputStream(out).use { ios ->
            writer.output = ios
            writer.write(null, javax.imageio.IIOImage(img, null, null), params)
        }
        writer.dispose()
        return ImageIO.read(ByteArrayInputStream(out.toByteArray()))
    }

    private fun channelDelta(a: Int, b: Int): Int = maxOf(
        abs(((a shr 16) and 0xFF) - ((b shr 16) and 0xFF)),
        abs(((a shr 8) and 0xFF) - ((b shr 8) and 0xFF)),
        abs((a and 0xFF) - (b and 0xFF)),
    )
}
