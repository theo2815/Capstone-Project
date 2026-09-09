package com.quickpitik.mobile.data.usb.ptp

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

data class PtpObjectInfo(
    val format: Int,
    val compressedSize: Long,
    val filename: String,
    val storageId: Long = 0,
    val parent: Long = 0,
    val capturedAt: String = "",
) {
    val isJpeg: Boolean get() = format == Ptp.OBJECT_FORMAT_EXIF_JPEG ||
        filename.substringAfterLast('.', "").lowercase() in setOf("jpg", "jpeg")
    val isAssociation: Boolean get() = format == Ptp.OBJECT_FORMAT_ASSOCIATION
    val expectedSize: Long? get() = compressedSize.takeIf { it in 1 until 0xffffffffL }

    // Metadata is a source locator, never proof of equal file contents.
    fun sourceKey(cameraKey: String, handle: Long): String = fingerprint(
        cameraKey, storageId, handle, parent, filename, compressedSize, format, capturedAt,
    )

    companion object {
        fun parse(data: ByteArray): PtpObjectInfo {
            if (data.size < 53) throw PtpException("Truncated camera object metadata.")
            val bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
            val storage = bb.getInt(0).toLong() and 0xffffffffL
            val format = bb.getShort(4).toInt() and 0xffff
            val size = bb.getInt(8).toLong() and 0xffffffffL
            val parent = bb.getInt(38).toLong() and 0xffffffffL
            bb.position(52)
            val filename = readString(bb)
            val capturedAt = if (bb.hasRemaining()) readString(bb) else ""
            return PtpObjectInfo(format, size, filename, storage, parent, capturedAt)
        }

        private fun readString(bb: ByteBuffer): String {
            val count = bb.get().toInt() and 0xff
            if (count > bb.remaining() / 2) throw PtpException("Truncated camera object string.")
            return buildString { repeat(count) { val ch = bb.short.toInt() and 0xffff; if (ch != 0) append(ch.toChar()) } }
        }
    }
}

internal fun fingerprint(vararg values: Any): String = MessageDigest.getInstance("SHA-256")
    .digest(values.joinToString("") { val text = it.toString(); "${text.length}:$text" }.toByteArray(Charsets.UTF_8))
    .joinToString("") { "%02x".format(it) }
