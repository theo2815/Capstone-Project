package com.quickpitik.service.photographer

import com.drew.imaging.ImageMetadataReader
import com.drew.metadata.exif.ExifIFD0Directory
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.awt.AlphaComposite
import java.awt.Color
import java.awt.Font
import java.awt.RenderingHints
import java.awt.geom.AffineTransform
import java.awt.image.AffineTransformOp
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO

@Service
class WatermarkService {
    private val log = LoggerFactory.getLogger(javaClass)

    // Single processed output that doubles as thumbnail + public watermark for
    // PR 7. Long-edge cap at 1280px keeps payload tiny while staying sharp on
    // the runner mosaic. JPEG re-encode normalizes content type so the FE wire
    // can assume image/jpeg regardless of upload format.
    //
    // Orientation: phone cameras (iPhone) save portrait photos as landscape
    // pixels + an EXIF orientation tag asking the viewer to rotate on display.
    // ImageIO.read drops EXIF, so without explicit rotation every portrait
    // upload renders sideways on the web. We read the tag from the original
    // bytes, then rotate the BufferedImage upright before resize + watermark.
    // The output JPEG carries no EXIF (re-encode strips it) but the pixels are
    // already in display orientation — clients render correctly without any
    // FE-side EXIF handling.
    fun processThumbnail(input: ByteArray, watermarkLabel: String): ByteArray {
        val source = ImageIO.read(ByteArrayInputStream(input))
            ?: throw IllegalArgumentException("Unreadable image bytes")
        val orientation = readExifOrientation(input)
        val upright = applyExifOrientation(source, orientation)
        val target = scaleToLongEdge(upright, MAX_LONG_EDGE)
        drawWatermark(target, watermarkLabel)
        val out = ByteArrayOutputStream()
        ImageIO.write(target, "jpg", out)
        return out.toByteArray()
    }

    private fun readExifOrientation(input: ByteArray): Int {
        return try {
            val metadata = ImageMetadataReader.readMetadata(ByteArrayInputStream(input))
            val dir = metadata.getFirstDirectoryOfType(ExifIFD0Directory::class.java)
                ?: return ORIENTATION_NORMAL
            if (!dir.containsTag(ExifIFD0Directory.TAG_ORIENTATION)) return ORIENTATION_NORMAL
            dir.getInt(ExifIFD0Directory.TAG_ORIENTATION)
        } catch (ex: Exception) {
            // PNG / WebP / EXIF-less JPEG / corrupt headers all land here. The
            // default (1 = normal) means the BufferedImage is returned as-is,
            // which is the correct fallback for any non-iPhone source.
            log.debug("EXIF orientation read failed; defaulting to 1: {}", ex.message)
            ORIENTATION_NORMAL
        }
    }

    private fun applyExifOrientation(source: BufferedImage, orientation: Int): BufferedImage {
        if (orientation <= ORIENTATION_NORMAL) return source
        val w = source.width
        val h = source.height
        val transform = AffineTransform()
        when (orientation) {
            2 -> { // horizontal flip
                transform.scale(-1.0, 1.0)
                transform.translate(-w.toDouble(), 0.0)
            }
            3 -> { // 180° rotation
                transform.translate(w.toDouble(), h.toDouble())
                transform.rotate(Math.PI)
            }
            4 -> { // vertical flip
                transform.scale(1.0, -1.0)
                transform.translate(0.0, -h.toDouble())
            }
            5 -> { // transpose: 90° CCW + horizontal flip
                transform.rotate(-Math.PI / 2)
                transform.scale(-1.0, 1.0)
            }
            6 -> { // 90° CW — by far the most common (iPhone portrait)
                transform.translate(h.toDouble(), 0.0)
                transform.rotate(Math.PI / 2)
            }
            7 -> { // transverse: 90° CW + horizontal flip
                transform.scale(-1.0, 1.0)
                transform.translate(-h.toDouble(), 0.0)
                transform.translate(0.0, w.toDouble())
                transform.rotate(3 * Math.PI / 2)
            }
            8 -> { // 90° CCW
                transform.translate(0.0, w.toDouble())
                transform.rotate(3 * Math.PI / 2)
            }
            else -> return source // unknown orientation tag, defensive default
        }
        // 90° and 270° rotations swap the output dimensions.
        val swapsDims = orientation in 5..8
        val outW = if (swapsDims) h else w
        val outH = if (swapsDims) w else h
        val out = BufferedImage(outW, outH, BufferedImage.TYPE_INT_RGB)
        val op = AffineTransformOp(transform, AffineTransformOp.TYPE_BILINEAR)
        op.filter(source, out)
        return out
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
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
            g.drawImage(source, 0, 0, targetW, targetH, null)
        } finally {
            g.dispose()
        }
        return resized
    }

    private fun drawWatermark(image: BufferedImage, label: String) {
        val text = label.trim().take(MAX_WATERMARK_CHARS).ifBlank { "QUICKPITIK" }
        val g = image.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
            val fontSize = (image.height * 0.035).toInt().coerceAtLeast(14)
            g.font = Font(Font.SANS_SERIF, Font.BOLD, fontSize)
            val metrics = g.fontMetrics
            val textWidth = metrics.stringWidth(text)
            val textHeight = metrics.height
            val pad = (fontSize * 0.6).toInt().coerceAtLeast(8)
            val x = image.width - textWidth - pad
            val y = image.height - pad
            // Translucent dark plate keeps the label readable on bright frames.
            g.composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.35f)
            g.color = Color.BLACK
            g.fillRoundRect(
                x - pad / 2,
                y - textHeight + metrics.descent / 2,
                textWidth + pad,
                textHeight,
                fontSize / 2,
                fontSize / 2,
            )
            g.composite = AlphaComposite.SrcOver
            g.color = Color(255, 255, 255, 220)
            g.drawString(text, x, y - metrics.descent / 2)
        } finally {
            g.dispose()
        }
    }

    companion object {
        private const val MAX_LONG_EDGE = 1280
        private const val MAX_WATERMARK_CHARS = 24
        private const val ORIENTATION_NORMAL = 1
    }
}
