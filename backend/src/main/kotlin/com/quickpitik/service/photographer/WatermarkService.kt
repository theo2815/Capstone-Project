package com.quickpitik.service.photographer

import com.quickpitik.service.image.ExifOrientation
import org.springframework.stereotype.Service
import java.awt.AlphaComposite
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
    //
    // Orientation: phone cameras (iPhone) save portrait photos as landscape
    // pixels + an EXIF orientation tag asking the viewer to rotate on display.
    // ImageIO.read drops EXIF, so without explicit rotation every portrait
    // upload renders sideways on the web. We read the tag from the original
    // bytes, then rotate the BufferedImage upright before resize + watermark.
    // The output JPEG carries no EXIF (re-encode strips it) but the pixels are
    // already in display orientation — clients render correctly without any
    // FE-side EXIF handling.
    //
    // Watermark is the photographer's uploaded logo image (PNG preserves
    // transparency, JPEG composites as a rectangle). Two marks per photo:
    //   1. Bottom-right corner — identification mark, sharp + visible.
    //   2. Diagonal center — anti-piracy mark, low opacity, harder to crop out.
    fun processThumbnail(input: ByteArray, watermarkImage: ByteArray): ByteArray {
        val source = ImageIO.read(ByteArrayInputStream(input))
            ?: throw IllegalArgumentException("Unreadable image bytes")
        val orientation = ExifOrientation.read(input)
        val upright = ExifOrientation.apply(source, orientation)
        val target = scaleToLongEdge(upright, MAX_LONG_EDGE)
        val watermark = ImageIO.read(ByteArrayInputStream(watermarkImage))
            ?: throw IllegalArgumentException("Unreadable watermark image bytes")
        drawWatermarkImage(target, watermark)
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

    private fun drawWatermarkImage(photo: BufferedImage, watermark: BufferedImage) {
        // Corner mark — identifies the photographer. ~15% of photo width,
        // padded from edges, 70% opacity. This is the "this photo is mine" mark.
        val cornerWidth = (photo.width * CORNER_WIDTH_RATIO).toInt().coerceAtLeast(48)
        val cornerScale = cornerWidth.toDouble() / watermark.width
        val cornerHeight = (watermark.height * cornerScale).toInt().coerceAtLeast(1)
        val cornerPad = (photo.width * 0.025).toInt().coerceAtLeast(12)

        val gCorner = photo.createGraphics()
        try {
            gCorner.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            gCorner.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
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

        // Center mark — diagonal across the photo, low opacity. This is the
        // anti-piracy mark: bigger so it's hard to crop out, low opacity so it
        // doesn't dominate the preview. Rotated -18° so it reads as a
        // watermark and not a misaligned logo.
        val centerWidth = (photo.width * CENTER_WIDTH_RATIO).toInt().coerceAtLeast(120)
        val centerScale = centerWidth.toDouble() / watermark.width
        val centerHeight = (watermark.height * centerScale).toInt().coerceAtLeast(1)

        val gCenter = photo.createGraphics()
        try {
            gCenter.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            gCenter.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
            gCenter.composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, CENTER_OPACITY)
            // Translate to photo center, rotate, then draw the watermark
            // centered at the origin — so the rotation pivot is the watermark's
            // own center, not a corner.
            gCenter.translate(photo.width / 2.0, photo.height / 2.0)
            gCenter.rotate(Math.toRadians(CENTER_ROTATION_DEG))
            gCenter.drawImage(
                watermark,
                -centerWidth / 2,
                -centerHeight / 2,
                centerWidth,
                centerHeight,
                null,
            )
        } finally {
            gCenter.dispose()
        }
    }

    companion object {
        private const val MAX_LONG_EDGE = 1280
        // Corner watermark — 15% of photo width, 70% opacity (sharp + visible).
        private const val CORNER_WIDTH_RATIO = 0.15
        private const val CORNER_OPACITY = 0.70f
        // Center diagonal watermark — 50% of photo width, 18% opacity, -18°.
        private const val CENTER_WIDTH_RATIO = 0.50
        private const val CENTER_OPACITY = 0.18f
        private const val CENTER_ROTATION_DEG = -18.0
    }
}
