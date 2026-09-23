package com.quickpitik.mobile.data.usb.ptp

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** One Canon EOS event record pulled from an EOS_GetEvent payload. */
data class EosEvent(val code: Int, val handle: Long) {
    val isObjectArrival: Boolean
        get() = code == Ptp.EOS_EVENT_OBJECT_ADDED_EX ||
            code == Ptp.EOS_EVENT_OBJECT_ADDED_EX64 ||
            code == Ptp.EOS_EVENT_OBJECT_ADDED_EX64_LFN ||
            code == Ptp.EOS_EVENT_REQUEST_OBJECT_TRANSFER
}

/** Parser for the packed record list inside an EOS_GetEvent payload. */
object EosEvents {
    /**
     * Each record is `[UINT32 size][UINT32 eventCode][payload…]`, laid back to
     * back. For the object-arrival events the new object handle is the first
     * field of the payload (offset +8 within the record). We only need the
     * handle — GetObject's own data-phase length tells us the byte count — so we
     * don't depend on the fiddlier per-firmware size/filename offsets.
     */
    fun parse(blob: ByteArray): List<EosEvent> {
        if (blob.size < 8) return emptyList()
        val bb = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN)
        val events = ArrayList<EosEvent>()
        var off = 0
        while (off + 8 <= blob.size) {
            val size = bb.getInt(off)
            if (size < 8) break // size 0 / terminator / malformed
            if (size > blob.size - off) throw PtpException("Truncated EOS event record.")
            val code = bb.getInt(off + 4)
            // size >= 12 keeps a payload-less (8-byte) record from reading the
            // NEXT record's size field as its handle.
            if (code != 0 && size >= 12 && off + 12 <= blob.size) {
                val handle = bb.getInt(off + 8).toLong() and 0xFFFFFFFFL
                events.add(EosEvent(code, handle))
            }
            off += size
        }
        return events
    }
}
