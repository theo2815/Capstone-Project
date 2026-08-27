package com.quickpitik.service.image

import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

// Decompression-bomb guard: reads only the image HEADER (no pixel decode) and
// rejects declared dimensions whose full decode would blow the heap — a 25 MB
// PNG may legally declare 50000×50000 (~10 GB as INT_ARGB), and even a 5 MB
// JPEG of flat color can declare 65500×65500. Call before any ImageIO.read of
// client-supplied bytes. Returns false when no reader claims the bytes — the
// subsequent full decode then produces the existing unreadable-image error.
object ImagePixelGuard {
    // 80 MP ≈ 320 MB decoded INT_ARGB — above any current camera (A7R V is
    // 61 MP) with headroom. ponytail: one global cap; split per endpoint only
    // if a real need appears.
    const val MAX_PIXELS = 80_000_000L

    // maxPixels is parameterized for tests (a real >80 MP fixture would be
    // absurd to ship); production call sites use the default.
    fun exceedsPixelBudget(bytes: ByteArray, maxPixels: Long = MAX_PIXELS): Boolean =
        ImageIO.createImageInputStream(ByteArrayInputStream(bytes)).use { stream ->
            val readers = ImageIO.getImageReaders(stream)
            if (!readers.hasNext()) return false
            val reader = readers.next()
            try {
                reader.input = stream
                reader.getWidth(0).toLong() * reader.getHeight(0).toLong() > maxPixels
            } catch (ex: Exception) {
                false
            } finally {
                reader.dispose()
            }
        }
}
