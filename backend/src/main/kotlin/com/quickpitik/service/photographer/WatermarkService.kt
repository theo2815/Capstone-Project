package com.quickpitik.service.photographer

import com.quickpitik.service.image.ExifOrientation
import com.quickpitik.service.image.JpegXmp
import com.quickpitik.service.image.PerceptualHash
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.awt.AlphaComposite
import java.awt.Color
import java.awt.Font
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.geom.Point2D
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.time.Year
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.math.hypot
import kotlin.random.Random

// Who took the photo, as baked into the preview: `name` is the studio/brand
// name (falls back to the account name), `handle` is null until verification,
// `photoId` seeds the per-photo tile jitter and goes into the XMP packet.
data class WatermarkCredit(val name: String, val handle: String?, val photoId: UUID)

// The preview JPEG plus its perceptual hash — computed from the same pixels
// before encoding so the caller never has to decode the output again.
data class MarkedPreview(val jpeg: ByteArray, val phash: Long)

@Service
class WatermarkService {

    private val log = LoggerFactory.getLogger(javaClass)

    // Single processed output that doubles as thumbnail + public preview. Long
    // edge capped at 1280px: sharp on the runner mosaic, never print-quality if
    // screenshotted. JPEG re-encode normalizes the content type; EXIF is
    // dropped (orientation is applied to the pixels first) and replaced by the
    // XMP credit packet.
    //
    // Three layers, bottom to top:
    //   1. QuickPitik credit tiles — wordmark rows alternating with
    //      "© Name · @handle · QuickPitik" rows across the WHOLE frame at low
    //      opacity, rotated, phase/angle jittered per photo, each tile drawn
    //      light or dark against what is under it. Cropping can't remove it,
    //      a fixed template can't be subtracted, and inpainting it away has
    //      to hallucinate across the runner.
    //   2. Crisp bottom-left caption with the same credit — the attribution
    //      that stays legible at phone size where the tiles are just texture.
    //   3. The photographer's own uploaded logo, corner + center, on top.
    fun processThumbnail(input: ByteArray, watermarkImage: ByteArray, credit: WatermarkCredit): MarkedPreview {
        val source = ImageIO.read(ByteArrayInputStream(input))
            ?: throw IllegalArgumentException("Unreadable image bytes")
        val upright = ExifOrientation.apply(source, ExifOrientation.read(input))
        val target = scaleToLongEdge(upright, MAX_LONG_EDGE)
        val watermark = ImageIO.read(ByteArrayInputStream(watermarkImage))
            ?: throw IllegalArgumentException("Unreadable watermark image bytes")
        drawCreditTiles(target, credit)
        drawCaption(target, credit)
        drawWatermarkImage(target, watermark)
        val phash = PerceptualHash.of(target)
        val out = ByteArrayOutputStream()
        ImageIO.write(target, "jpg", out)
        val packet = JpegXmp.creditPacket(credit.name, credit.handle, credit.photoId, Year.now().value)
        return MarkedPreview(JpegXmp.inject(out.toByteArray(), packet), phash)
    }

    private fun scaleToLongEdge(source: BufferedImage, maxLongEdge: Int): BufferedImage {
        val srcW = source.width
        val srcH = source.height
        val longEdge = maxOf(srcW, srcH)
        val scale = if (longEdge > maxLongEdge) maxLongEdge.toDouble() / longEdge else 1.0
        val targetW = (srcW * scale).toInt().coerceAtLeast(1)
        val targetH = (srcH * scale).toInt().coerceAtLeast(1)
        val resized = BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_RGB)
        val g = resized.createGraphics()
        try {
            hints(g)
            g.drawImage(source, 0, 0, targetW, targetH, null)
        } finally {
            g.dispose()
        }
        return resized
    }

    private fun drawCreditTiles(photo: BufferedImage, credit: WatermarkCredit) {
        val w = photo.width
        val h = photo.height
        val rnd = Random(credit.photoId.mostSignificantBits xor credit.photoId.leastSignificantBits)
        val angle = Math.toRadians(BASE_ROTATION_DEG + (rnd.nextDouble() * 2 - 1) * ROTATION_JITTER_DEG)
        val markW = (w * TILE_WIDTH_RATIO).toInt().coerceAtLeast(96)
        val markH = (markW * wordmarkLight.height / wordmarkLight.width.toDouble()).toInt().coerceAtLeast(1)
        val line = creditLine(credit)

        val g = photo.createGraphics()
        try {
            hints(g)
            g.composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, TILE_OPACITY)
            g.translate(w / 2.0, h / 2.0)
            g.rotate(angle)
            val text = textMetrics(g, (w * TILE_TEXT_RATIO).toFloat().coerceIn(14f, 40f), line)
            val rowPitch = (maxOf(markH, text?.height ?: 0) * 1.4).toInt().coerceAtLeast(1)
            val reach = (hypot(w.toDouble(), h.toDouble()) / 2).toInt()
            val phaseX = rnd.nextInt(markW)
            val phaseY = rnd.nextInt(rowPitch)
            var row = 0
            var y = -reach - rowPitch + phaseY
            while (y <= reach + rowPitch) {
                val textRow = row % 2 == 1 && text != null
                val pitchX = if (textRow) text!!.width + markW / 2 else (markW * TILE_GAP_RATIO).toInt()
                var x = -reach - pitchX + phaseX + if (row % 2 == 1) pitchX / 2 else 0
                while (x <= reach + pitchX) {
                    val bright = isBrightUnder(photo, g, x + pitchX / 2.0, y + rowPitch / 2.0)
                    if (textRow) {
                        g.color = if (bright) INK else Color.WHITE
                        g.drawString(line, x, y + text!!.ascent)
                    } else {
                        g.drawImage(if (bright) wordmarkDark else wordmarkLight, x, y, markW, markH, null)
                    }
                    x += pitchX
                }
                y += rowPitch
                row++
            }
        } finally {
            g.dispose()
        }
    }

    private fun drawCaption(photo: BufferedImage, credit: WatermarkCredit) {
        val w = photo.width
        val h = photo.height
        val pad = (w * 0.025).toInt().coerceAtLeast(12)
        // Leave the bottom-right free for the photographer's corner logo.
        val budget = w - (w * CORNER_WIDTH_RATIO).toInt().coerceAtLeast(48) - 3 * pad
        val g = photo.createGraphics()
        try {
            hints(g)
            val size = (w * CAPTION_TEXT_RATIO).toFloat().coerceIn(16f, 44f)
            var line = creditLine(credit)
            var text = textMetrics(g, size, line) ?: return
            if (text.width > budget) {
                line = creditLine(credit.copy(handle = null))
                text = textMetrics(g, size, line) ?: return
            }
            var name = credit.name
            while (text.width > budget && name.length > 4) {
                name = name.dropLast(1)
                line = creditLine(WatermarkCredit(name.trimEnd() + "…", null, credit.photoId))
                text = textMetrics(g, size, line) ?: return
            }
            val baseline = h - pad - text.descent
            val bright = isBrightUnder(photo, g, pad + text.width / 2.0, baseline - text.ascent / 2.0)
            g.composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, CAPTION_OPACITY)
            // 1px shadow in the opposite polarity keeps the caption readable
            // where the sample window straddles a light/dark edge.
            g.color = if (bright) Color.WHITE else INK
            g.drawString(line, pad + 1, baseline + 1)
            g.color = if (bright) INK else Color.WHITE
            g.drawString(line, pad, baseline)
        } finally {
            g.dispose()
        }
    }

    private fun drawWatermarkImage(photo: BufferedImage, watermark: BufferedImage) {
        // Corner mark — identifies the photographer. ~15% of photo width,
        // padded from edges, 70% opacity. This is the "this photo is mine" mark.
        val cornerWidth = (photo.width * CORNER_WIDTH_RATIO).toInt().coerceAtLeast(48)
        val cornerScale = cornerWidth.toDouble() / watermark.width
        val cornerHeight = (watermark.height * cornerScale).toInt().coerceAtLeast(1)
        val cornerPad = (photo.width * 0.025).toInt().coerceAtLeast(12)

        val gCorner = photo.createGraphics()
        try {
            hints(gCorner)
            gCorner.composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, CORNER_OPACITY)
            gCorner.drawImage(
                watermark,
                photo.width - cornerWidth - cornerPad,
                photo.height - cornerHeight - cornerPad,
                cornerWidth,
                cornerHeight,
                null,
            )
        } finally {
            gCorner.dispose()
        }

        // Center mark — diagonal across the photo, low opacity. Bigger so it's
        // hard to crop out, low opacity so it doesn't dominate the preview.
        val centerWidth = (photo.width * CENTER_WIDTH_RATIO).toInt().coerceAtLeast(120)
        val centerScale = centerWidth.toDouble() / watermark.width
        val centerHeight = (watermark.height * centerScale).toInt().coerceAtLeast(1)

        val gCenter = photo.createGraphics()
        try {
            hints(gCenter)
            gCenter.composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, CENTER_OPACITY)
            gCenter.translate(photo.width / 2.0, photo.height / 2.0)
            gCenter.rotate(Math.toRadians(BASE_ROTATION_DEG))
            gCenter.drawImage(watermark, -centerWidth / 2, -centerHeight / 2, centerWidth, centerHeight, null)
        } finally {
            gCenter.dispose()
        }
    }

    private fun creditLine(c: WatermarkCredit): String =
        listOfNotNull("© ${c.name.take(40)}", c.handle?.let { "@$it" }, "QuickPitik").joinToString(" · ")

    private class TextMetrics(val width: Int, val height: Int, val ascent: Int, val descent: Int)

    // Sets the font on `g` and measures `line`. Null when no font can be used
    // — the tiles then carry the wordmark only. A JDK image without
    // fontconfig/freetype is the realistic way to get here (see
    // backend/CLAUDE.md common issues); logged once, never fatal.
    private fun textMetrics(g: Graphics2D, size: Float, line: String): TextMetrics? {
        val font = creditFont ?: return null
        if (textDisabled) return null
        return try {
            g.font = font.deriveFont(size)
            val fm = g.fontMetrics
            TextMetrics(fm.stringWidth(line), fm.height, fm.ascent, fm.descent)
        } catch (ex: Throwable) {
            textDisabled = true
            log.warn("Credit text disabled — the JVM cannot rasterize fonts ({}); previews carry wordmark tiles only", ex.toString())
            null
        }
    }

    // Mean luminance of a 5×5 window (stride 4px) around a point given in the
    // graphics' current user space. Drives light-on-dark / dark-on-light per
    // tile so the mark is visible on every background and no single template
    // matches the whole layer.
    // ponytail: center-window sample; a tile straddling a light/dark edge picks one polarity.
    private fun isBrightUnder(photo: BufferedImage, g: Graphics2D, ux: Double, uy: Double): Boolean {
        val p = g.transform.transform(Point2D.Double(ux, uy), null)
        var sum = 0L
        for (dy in -2..2) for (dx in -2..2) {
            val x = (p.x + dx * 4).toInt().coerceIn(0, photo.width - 1)
            val y = (p.y + dy * 4).toInt().coerceIn(0, photo.height - 1)
            val rgb = photo.getRGB(x, y)
            sum += (299 * ((rgb shr 16) and 0xFF) + 587 * ((rgb shr 8) and 0xFF) + 114 * (rgb and 0xFF)) / 1000
        }
        return sum / 25 > 128
    }

    private fun hints(g: Graphics2D) {
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
    }

    // The brand wordmark, recoloured through its own alpha so one asset gives
    // both polarities. Loaded once per JVM.
    private val wordmarkLight: BufferedImage by lazy { tinted(Color.WHITE) }
    private val wordmarkDark: BufferedImage by lazy { tinted(INK) }

    private fun tinted(color: Color): BufferedImage {
        val src = javaClass.getResourceAsStream(WORDMARK_RESOURCE)?.use { ImageIO.read(it) }
            ?: error("Missing brand asset $WORDMARK_RESOURCE")
        val out = BufferedImage(src.width, src.height, BufferedImage.TYPE_INT_ARGB)
        val rgb = color.rgb and 0xFFFFFF
        for (y in 0 until src.height) for (x in 0 until src.width) {
            out.setRGB(x, y, (src.getRGB(x, y) and 0xFF000000.toInt()) or rgb)
        }
        return out
    }

    @Volatile
    private var textDisabled = false

    private val creditFont: Font? by lazy {
        try {
            javaClass.getResourceAsStream(FONT_RESOURCE)?.use { Font.createFont(Font.TRUETYPE_FONT, it) }
                ?: error("Missing font asset $FONT_RESOURCE")
        } catch (ex: Throwable) {
            log.warn("Bundled credit font unavailable ({}); falling back to the JVM sans-serif", ex.toString())
            try {
                Font(Font.SANS_SERIF, Font.BOLD, 12)
            } catch (ex2: Throwable) {
                log.warn("No usable font ({}); previews carry wordmark tiles only", ex2.toString())
                null
            }
        }
    }

    companion object {
        private const val MAX_LONG_EDGE = 1280
        private const val WORDMARK_RESOURCE = "/brand/quickpitik-wordmark.png"
        private const val FONT_RESOURCE = "/fonts/Archivo-SemiBold.ttf"
        private val INK = Color(0x11, 0x11, 0x11)

        // QuickPitik credit tiles — wordmark ~22% of width, rows alternate
        // wordmark / credit text, 25% opacity, -18° ± 4° per photo.
        private const val TILE_WIDTH_RATIO = 0.22
        private const val TILE_GAP_RATIO = 1.6
        private const val TILE_TEXT_RATIO = 0.024
        private const val TILE_OPACITY = 0.25f
        private const val BASE_ROTATION_DEG = -18.0
        private const val ROTATION_JITTER_DEG = 4.0
        // Bottom-left credit caption — the legible-at-phone-size attribution.
        private const val CAPTION_TEXT_RATIO = 0.028
        private const val CAPTION_OPACITY = 0.85f
        // Photographer's logo — corner 15% of photo width, 70% opacity; center
        // diagonal 50% of width, 18% opacity. Unchanged since 2026-05-18.
        private const val CORNER_WIDTH_RATIO = 0.15
        private const val CORNER_OPACITY = 0.70f
        private const val CENTER_WIDTH_RATIO = 0.50
        private const val CENTER_OPACITY = 0.18f
    }
}
