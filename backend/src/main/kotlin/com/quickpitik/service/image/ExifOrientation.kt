package com.quickpitik.service.image

import com.drew.imaging.ImageMetadataReader
import com.drew.metadata.exif.ExifIFD0Directory
import org.slf4j.LoggerFactory
import java.awt.RenderingHints
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream

/**
 * EXIF orientation handling, shared by the two upload paths that need pixels in
 * display orientation before anything downstream reads them:
 *
 *  - `WatermarkService` — photographer photo uploads (composite + re-encode).
 *  - `SelfieService` — runner selfies forwarded to ai-api for face matching.
 *
 * Phone cameras (iPhone especially) store portrait shots as landscape pixels plus
 * an EXIF tag asking the viewer to rotate on display. `ImageIO.read` drops EXIF,
 * so without explicit rotation a portrait upload stays sideways — which renders
 * wrong on the web and makes face detection miss the runner entirely.
 *
 * Stateless object rather than a `@Service`: it has no dependencies, and callers
 * in two different packages would otherwise both need constructor injection for
 * what is pure image math.
 */
object ExifOrientation {
    /** Orientation 1 — pixels are already upright; every code path treats this as "no work". */
    const val NORMAL = 1

    private val log = LoggerFactory.getLogger(ExifOrientation::class.java)

    /**
     * Reads the EXIF orientation tag from the original bytes, defaulting to
     * [NORMAL] when the tag is absent or unreadable.
     */
    fun read(input: ByteArray): Int {
        return try {
            val metadata = ImageMetadataReader.readMetadata(ByteArrayInputStream(input))
            val dir = metadata.getFirstDirectoryOfType(ExifIFD0Directory::class.java)
                ?: return NORMAL
            if (!dir.containsTag(ExifIFD0Directory.TAG_ORIENTATION)) return NORMAL
            dir.getInt(ExifIFD0Directory.TAG_ORIENTATION)
        } catch (ex: Exception) {
            // PNG / WebP / EXIF-less JPEG / corrupt headers all land here. The
            // default (1 = normal) means the BufferedImage is returned as-is,
            // which is the correct fallback for any non-iPhone source.
            log.debug("EXIF orientation read failed; defaulting to 1: {}", ex.message)
            NORMAL
        }
    }

    /**
     * Returns [source] rotated/flipped upright for the given [orientation].
     * Returns [source] unchanged for [NORMAL] and for any unknown tag value.
     */
    fun apply(source: BufferedImage, orientation: Int): BufferedImage {
        if (orientation <= NORMAL) return source
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
        // AffineTransformOp.filter throws ImagingOpException when source and
        // destination ColorModels disagree — common for ImageIO-loaded JPEGs
        // that come back as TYPE_3BYTE_BGR / TYPE_CUSTOM (iPhone, cameras with
        // embedded ICC profiles). Graphics2D.drawImage handles the conversion
        // gracefully and matches the scaling path's approach.
        val g = out.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            g.drawImage(source, transform, null)
        } finally {
            g.dispose()
        }
        return out
    }
}
