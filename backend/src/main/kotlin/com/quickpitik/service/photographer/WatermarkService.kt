package com.quickpitik.service.photographer

import org.springframework.stereotype.Service
import java.awt.AlphaComposite
import java.awt.Color
import java.awt.Font
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO

@Service
class WatermarkService {
    // Single processed output that doubles as thumbnail + public watermark for
    // PR 7. Long-edge cap at 1280px keeps payload tiny while staying sharp on
    // the runner mosaic. JPEG re-encode normalizes content type so the FE wire
    // can assume image/jpeg regardless of upload format.
    fun processThumbnail(input: ByteArray, watermarkLabel: String): ByteArray {
        val source = ImageIO.read(ByteArrayInputStream(input))
            ?: throw IllegalArgumentException("Unreadable image bytes")
        val target = scaleToLongEdge(source, MAX_LONG_EDGE)
        drawWatermark(target, watermarkLabel)
        val out = ByteArrayOutputStream()
        ImageIO.write(target, "jpg", out)
        return out.toByteArray()
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
    }
}
