package com.quickpitik.service.image

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ApiException
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Component
import java.io.ByteArrayInputStream
import java.io.InputStream
import java.security.MessageDigest
import java.util.HexFormat
import javax.imageio.ImageIO

@Component
class PhotoUploadValidator {
    fun canDecode(contentType: String): Boolean =
        ImageIO.getImageReadersByMIMEType(contentType.lowercase()).hasNext()

    fun validate(
        input: InputStream,
        contentType: String,
        expectedSize: Long,
        expectedHash: String? = null,
    ): String {
        val type = contentType.lowercase()
        if (type !in ALLOWED_CONTENT_TYPES) unsupported()

        val bytes = input.use { it.readNBytes(MAX_BYTES + 1) }
        if (bytes.isEmpty() || bytes.size > MAX_BYTES || bytes.size.toLong() != expectedSize) {
            throw ApiException(
                HttpStatus.PAYLOAD_TOO_LARGE,
                ErrorCodes.PAYLOAD_TOO_LARGE,
                "The stored image size does not match the upload.",
            )
        }

        val hash = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))
        if (expectedHash != null && hash != expectedHash.lowercase()) {
            throw ApiException(
                HttpStatus.UNPROCESSABLE_ENTITY,
                ErrorCodes.UPLOAD_CHECKSUM_MISMATCH,
                "The uploaded file checksum does not match. Upload it again.",
            )
        }

        validateImage(bytes, type)
        return hash
    }

    private fun validateImage(bytes: ByteArray, declaredType: String) {
        try {
            val stream = ImageIO.createImageInputStream(ByteArrayInputStream(bytes)) ?: unsupported()
            stream.use {
                val readers = ImageIO.getImageReaders(it)
                if (!readers.hasNext()) unsupported()
                val reader = readers.next()
                try {
                    reader.input = it
                    val actualType = when (reader.formatName.lowercase()) {
                        "jpg", "jpeg" -> "image/jpeg"
                        "png" -> "image/png"
                        "webp" -> "image/webp"
                        else -> unsupported()
                    }
                    if (actualType != declaredType) unsupported()

                    val width = reader.getWidth(0)
                    val height = reader.getHeight(0)
                    if (width <= 0 || height <= 0 || width.toLong() * height.toLong() > ImagePixelGuard.MAX_PIXELS) {
                        unsupported("Image dimensions exceed the supported maximum.")
                    }

                    val factor = SubsampledDecode.subsampleFactor(width, height, VALIDATION_LONG_EDGE)
                    val image = reader.read(
                        0,
                        reader.defaultReadParam.apply { setSourceSubsampling(factor, factor, 0, 0) },
                    ) ?: unsupported()
                    image.flush()
                } finally {
                    reader.dispose()
                }
            }
        } catch (error: ApiException) {
            throw error
        } catch (_: Exception) {
            unsupported()
        }
    }

    private fun unsupported(message: String = "Upload must be a readable JPEG, PNG, or WebP image."): Nothing =
        throw ApiException(HttpStatus.UNSUPPORTED_MEDIA_TYPE, ErrorCodes.UNSUPPORTED_MEDIA_TYPE, message)

    companion object {
        const val MAX_BYTES = 25 * 1024 * 1024
        private const val VALIDATION_LONG_EDGE = 2_048
        private val ALLOWED_CONTENT_TYPES = setOf("image/jpeg", "image/png", "image/webp")
    }
}
