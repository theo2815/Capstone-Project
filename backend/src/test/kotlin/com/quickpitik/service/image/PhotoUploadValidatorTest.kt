package com.quickpitik.service.image

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ApiException
import org.junit.jupiter.api.Test
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.security.MessageDigest
import java.util.Base64
import java.util.HexFormat
import java.util.zip.CRC32
import javax.imageio.ImageIO
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PhotoUploadValidatorTest {
    private val validator = PhotoUploadValidator()

    @Test
    fun `valid image returns its verified hash`() {
        val bytes = jpeg()
        val hash = sha256(bytes)

        val result = validator.validate(ByteArrayInputStream(bytes), "image/jpeg", bytes.size.toLong(), hash)

        assertEquals(hash, result)
    }

    @Test
    fun `runtime decoder support is reported before direct upload issuance`() {
        assertTrue(validator.canDecode("image/jpeg"))
        assertTrue(validator.canDecode("image/png"))
        assertFalse(validator.canDecode("image/webp"))
    }

    @Test
    fun `valid WebP remains unsupported for event photo validation`() {
        val bytes = Base64.getDecoder().decode(
            "UklGRjIAAABXRUJQVlA4ICYAAACQAQCdASoQABAAAUAmJYgCdLoAA5gA/vPfZrQtCBz+N5vNasQAAA==",
        )

        val error = assertFailsWith<ApiException> {
            validator.validate(ByteArrayInputStream(bytes), "image/webp", bytes.size.toLong())
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, error.code)
    }

    @Test
    fun `declared type must match the actual image format`() {
        val bytes = jpeg()

        val error = assertFailsWith<ApiException> {
            validator.validate(ByteArrayInputStream(bytes), "image/png", bytes.size.toLong())
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, error.code)
    }

    @Test
    fun `unreadable bytes are rejected`() {
        val bytes = "not an image".toByteArray()

        val error = assertFailsWith<ApiException> {
            validator.validate(ByteArrayInputStream(bytes), "image/jpeg", bytes.size.toLong())
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, error.code)
    }

    @Test
    fun `recorded size and maximum size are enforced from the stream`() {
        val bytes = jpeg()
        val atLimit = ByteArray(PhotoUploadValidator.MAX_BYTES).also { bytes.copyInto(it) }
        assertEquals(
            HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(atLimit)),
            validator.validate(ByteArrayInputStream(atLimit), "image/jpeg", atLimit.size.toLong()),
        )
        val mismatch = assertFailsWith<ApiException> {
            validator.validate(ByteArrayInputStream(bytes), "image/jpeg", bytes.size.toLong() + 1)
        }
        val overLimit = assertFailsWith<ApiException> {
            validator.validate(
                ByteArrayInputStream(ByteArray(PhotoUploadValidator.MAX_BYTES + 1)),
                "image/jpeg",
                (PhotoUploadValidator.MAX_BYTES + 1).toLong(),
            )
        }

        assertEquals(ErrorCodes.PAYLOAD_TOO_LARGE, mismatch.code)
        assertEquals(ErrorCodes.PAYLOAD_TOO_LARGE, overLimit.code)
    }

    @Test
    fun `declared checksum must match the original bytes`() {
        val bytes = jpeg()

        val error = assertFailsWith<ApiException> {
            validator.validate(ByteArrayInputStream(bytes), "image/jpeg", bytes.size.toLong(), "0".repeat(64))
        }

        assertEquals(ErrorCodes.UPLOAD_CHECKSUM_MISMATCH, error.code)
    }

    @Test
    fun `image headers above the pixel budget are rejected before decode`() {
        val bytes = pngHeader(width = 10_000, height = 10_000)

        val error = assertFailsWith<ApiException> {
            validator.validate(ByteArrayInputStream(bytes), "image/png", bytes.size.toLong())
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, error.code)
    }

    private fun jpeg(): ByteArray = ByteArrayOutputStream().use { output ->
        check(ImageIO.write(BufferedImage(32, 24, BufferedImage.TYPE_INT_RGB), "jpg", output))
        output.toByteArray()
    }

    private fun sha256(bytes: ByteArray): String =
        HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))

    private fun pngHeader(width: Int, height: Int): ByteArray {
        val output = ByteArrayOutputStream()
        output.write(byteArrayOf(-119, 80, 78, 71, 13, 10, 26, 10))
        val ihdr = ByteArrayOutputStream().apply {
            writeInt(width)
            writeInt(height)
            write(byteArrayOf(8, 2, 0, 0, 0))
        }.toByteArray()
        output.writeChunk("IHDR", ihdr)
        output.writeChunk("IEND", byteArrayOf())
        return output.toByteArray()
    }

    private fun ByteArrayOutputStream.writeChunk(type: String, data: ByteArray) {
        writeInt(data.size)
        val typed = type.toByteArray(Charsets.US_ASCII) + data
        write(typed)
        val crc = CRC32().apply { update(typed) }.value
        writeInt(crc.toInt())
    }

    private fun ByteArrayOutputStream.writeInt(value: Int) {
        write(byteArrayOf((value ushr 24).toByte(), (value ushr 16).toByte(), (value ushr 8).toByte(), value.toByte()))
    }
}
