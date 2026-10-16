package com.quickpitik.service.events

import com.quickpitik.exception.ApiException
import org.springframework.http.HttpStatus
import java.util.HexFormat
import java.security.MessageDigest
import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ValidationException
import com.quickpitik.service.image.ImageDecode
import com.quickpitik.service.image.ImagePixelGuard
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.util.UUID
import javax.imageio.ImageIO

/**
 * Server-side cover image pipeline for events. Mirrors AvatarService:
 * the client is the source of *intent* but the server is the source of
 * *truth* — every upload is decoded, center-cropped to 4:3, downscaled
 * to a max of 1920×1440, and re-encoded as JPEG before it touches
 * storage. That makes the public /events tile (and /events/[slug] hero)
 * predictable regardless of what the photographer's phone produced.
 *
 * Returns the storage key. The mapper presigns when serving the row,
 * matching the avatar TTL pattern.
 */
@Service
class EventCoverService(
    private val storageService: StorageService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun upload(eventId: UUID, file: ByteArray, contentType: String?): String =
        uploadPrepared("events/$eventId/cover/${UUID.randomUUID()}.jpg", prepare(file, contentType))

    fun prepare(file: ByteArray, contentType: String?): ByteArray {
        if (file.isEmpty()) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "cover file is empty",
                field = "cover",
            )
        }
        if (file.size > MAX_COVER_BYTES) {
            throw ValidationException(
                code = ErrorCodes.PAYLOAD_TOO_LARGE,
                message = "cover must be ≤ 8 MB",
                field = "cover",
            )
        }
        val mime = (contentType ?: "").lowercase()
        if (mime !in SUPPORTED_TYPES) {
            throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "cover must be jpeg, png, or webp",
                field = "cover",
            )
        }
        if (ImagePixelGuard.exceedsPixelBudget(file)) {
            throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "Image dimensions exceed the supported maximum.",
                field = "cover",
            )
        }
        return normaliseToJpeg(file)
    }

    fun digest(bytes: ByteArray): String = HexFormat.of().formatHex(
        MessageDigest.getInstance("SHA-256").digest(bytes))

    fun uploadPrepared(key: String, processed: ByteArray): String {
        storageService.put(key, processed, "image/jpeg")
        return key
    }

    fun recover(key: String, processed: ByteArray, digest: String): String {
        val metadata = storageService.metadata(key) ?: return uploadPrepared(key, processed)
        if (metadata.sizeBytes != processed.size.toLong() || metadata.contentType != "image/jpeg" ||
            digest(storageService.getBytes(key)) != digest) {
            throw ApiException(HttpStatus.CONFLICT,
                ErrorCodes.ADMIN_COMMAND_RECOVERY_REQUIRED, "Cover object does not match its command; administrative reconciliation is required")
        }
        return key
    }

    fun delete(key: String) {
        runCatching { storageService.delete(key) }
            .onFailure { log.warn("cover delete of {} failed: {}", key, it.message) }
    }

    private fun normaliseToJpeg(input: ByteArray): ByteArray {
        val original: BufferedImage = ImageDecode.readOrNull(input)
            ?: throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "image cannot be decoded",
                field = "cover",
            )
        val w = original.width
        val h = original.height
        // A source smaller than the target ratio itself has no 4:3 crop. The
        // integer division below floors to zero and getSubimage throws
        // RasterFormatException (1×1 source) — or, for a source that clears
        // the crop but not the downscale, BufferedImage throws
        // IllegalArgumentException (4×1 source). Both surfaced as a 500.
        // w ≥ 4 and h ≥ 3 makes every crop and output dimension ≥ 3 on both
        // branches, so neither can degenerate.
        if (w < TARGET_W || h < TARGET_H) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "cover must be at least ${TARGET_W}×${TARGET_H} px",
                field = "cover",
            )
        }
        // Center-crop to a 4:3 aspect ratio. Wider sources lose horizontal
        // margins; taller sources lose vertical margins. Same approach the
        // FE preview uses, so the photographer sees what runners see.
        val (cropW, cropH) = if (w * TARGET_H >= h * TARGET_W) {
            // Source is wider than 4:3 → crop horizontally.
            (h * TARGET_W / TARGET_H) to h
        } else {
            // Source is narrower than 4:3 → crop vertically.
            w to (w * TARGET_H / TARGET_W)
        }
        val cx = (w - cropW) / 2
        val cy = (h - cropH) / 2
        val cropped = original.getSubimage(cx, cy, cropW, cropH)
        // Downscale only — never upscale tiny sources, JPEG artifacting at
        // 4× scale looks worse than a small clean tile.
        val outW = minOf(cropW, MAX_WIDTH)
        val outH = outW * TARGET_H / TARGET_W
        val output = BufferedImage(outW, outH, BufferedImage.TYPE_INT_RGB)
        val g = output.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
            g.drawImage(cropped, 0, 0, outW, outH, null)
        } finally {
            g.dispose()
        }
        val out = ByteArrayOutputStream()
        ImageIO.write(output, "jpeg", out)
        return out.toByteArray()
    }

    private companion object {
        const val MAX_WIDTH = 1920
        const val TARGET_W = 4
        const val TARGET_H = 3
        const val MAX_COVER_BYTES = 8 * 1024 * 1024
        val SUPPORTED_TYPES = setOf("image/jpeg", "image/jpg", "image/png", "image/webp")
    }
}
