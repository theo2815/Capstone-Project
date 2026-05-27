package com.quickpitik.mobile.data.usb.ptp

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The standard PTP ObjectInfo dataset (ISO 15740 §5.5.2), as returned by
 * GetObjectInfo. The card-poll path reads this for each new object handle on the
 * card to decide what to pull: only the JPEG, skipping folders (associations)
 * and RAW.
 *
 * Only the three fields we use are parsed:
 *  - ObjectFormat        — UINT16 at offset 4
 *  - ObjectCompressedSize— UINT32 at offset 8
 *  - Filename            — a PTP string at the fixed offset 52 (after the 13
 *                          fixed fields: StorageID/Format/Protection/Size/
 *                          Thumb×4/Image×3/BitDepth/Parent/Assoc×2/Sequence).
 */
data class PtpObjectInfo(
    val format: Int,
    val compressedSize: Long,
    val filename: String,
) {
    /** A JPEG the camera wrote — what we upload (RAW is left on the card). */
    val isJpeg: Boolean
        get() = format == Ptp.OBJECT_FORMAT_EXIF_JPEG ||
            filename.substringAfterLast('.', "").lowercase() in JPEG_EXT

    /** A folder/association, never a file to upload. */
    val isAssociation: Boolean
        get() = format == Ptp.OBJECT_FORMAT_ASSOCIATION

    companion object {
        private const val FILENAME_OFFSET = 52
        private val JPEG_EXT = setOf("jpg", "jpeg")

        fun parse(data: ByteArray): PtpObjectInfo {
            val bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
            val format = if (data.size >= 6) bb.getShort(4).toInt() and 0xFFFF else 0
            val size = if (data.size >= 12) bb.getInt(8).toLong() and 0xFFFFFFFFL else 0L
            return PtpObjectInfo(format, size, parsePtpString(data, FILENAME_OFFSET))
        }

        /**
         * A PTP string: a UINT8 length (number of UTF-16 code units INCLUDING the
         * trailing null), then that many little-endian UTF-16 chars. Length 0 is
         * an empty string.
         */
        private fun parsePtpString(data: ByteArray, offset: Int): String {
            if (offset >= data.size) return ""
            val count = data[offset].toInt() and 0xFF
            if (count == 0) return ""
            val sb = StringBuilder(count)
            var p = offset + 1
            // count includes the null terminator — stop one short of it.
            repeat(count - 1) {
                if (p + 1 >= data.size) return@repeat
                val ch = (data[p].toInt() and 0xFF) or ((data[p + 1].toInt() and 0xFF) shl 8)
                if (ch != 0) sb.append(ch.toChar())
                p += 2
            }
            return sb.toString()
        }
    }
}
