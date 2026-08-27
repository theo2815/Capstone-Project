package com.quickpitik.mobile.data

import java.io.ByteArrayOutputStream
import java.io.InputStream

// The one upload cap for user-picked images (avatar, cover, watermark, payout
// QR, selfies). Mirrors the website's MAX_UPLOAD_BYTES in
// website/src/lib/image-utils.ts — keep the two in lockstep.
internal const val MAX_UPLOAD_BYTES = 8 * 1024 * 1024

internal fun InputStream.readAtMost(byteCount: Int): ByteArray {
    require(byteCount >= 0)
    val output = ByteArrayOutputStream(minOf(byteCount, DEFAULT_BUFFER_SIZE))
    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
    var remaining = byteCount
    while (remaining > 0) {
        val read = read(buffer, 0, minOf(buffer.size, remaining))
        if (read < 0) break
        output.write(buffer, 0, read)
        remaining -= read
    }
    return output.toByteArray()
}
