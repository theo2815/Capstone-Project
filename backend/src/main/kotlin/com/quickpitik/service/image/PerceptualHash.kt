package com.quickpitik.service.image

import java.awt.RenderingHints
import java.awt.image.BufferedImage
import kotlin.math.PI
import kotlin.math.cos

// 64-bit perceptual hash (the classic DCT pHash) of an image: what the picture
// coarsely *looks like*, not what its bytes are. Two renderings of the same
// photo — screenshot, JPEG re-save, resize — land within a few bits of each
// other; a different photo lands ~32 bits away. Stored per preview in
// photos.phash (V42) and matched by Hamming distance in
// PhotoRepository.findNearestByPhash.
object PerceptualHash {

    fun of(img: BufferedImage): Long {
        val gray = grayscale32(img)
        val dct = dct2(gray)
        // Low-frequency 8×8 corner, DC term excluded from both the median and
        // the bits — it only encodes overall brightness.
        val coeffs = DoubleArray(LOW * LOW) { i -> dct[(i / LOW) * SIZE + (i % LOW)] }
        val median = coeffs.copyOfRange(1, coeffs.size).sorted().let { it[it.size / 2] }
        var bits = 0L
        for (i in 1 until coeffs.size) if (coeffs[i] > median) bits = bits or (1L shl i)
        return bits
    }

    fun distance(a: Long, b: Long): Int = java.lang.Long.bitCount(a xor b)

    // Two-step shrink: bilinear to 256² (cheap, bounded work for any input
    // size), then an exact 8×8 box average to 32². The box step is what makes
    // the hash stable — a straight bilinear jump to 32² point-samples, so a
    // one-pixel shift in a screenshot would flip bits.
    private fun grayscale32(img: BufferedImage): DoubleArray {
        val mid = BufferedImage(MID, MID, BufferedImage.TYPE_BYTE_GRAY)
        val g = mid.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            g.drawImage(img, 0, 0, MID, MID, null)
        } finally {
            g.dispose()
        }
        val raster = mid.raster
        val box = MID / SIZE
        val out = DoubleArray(SIZE * SIZE)
        for (y in 0 until SIZE) for (x in 0 until SIZE) {
            var sum = 0
            for (dy in 0 until box) for (dx in 0 until box) {
                sum += raster.getSample(x * box + dx, y * box + dy, 0)
            }
            out[y * SIZE + x] = sum.toDouble() / (box * box)
        }
        return out
    }

    // Separable DCT-II over the 32×32 grid: rows, then columns. ~65k multiplies
    // — no need for anything cleverer at this size.
    private fun dct2(px: DoubleArray): DoubleArray {
        val rows = DoubleArray(SIZE * SIZE)
        for (y in 0 until SIZE) dct1(px, y * SIZE, 1, rows, y * SIZE, 1)
        val out = DoubleArray(SIZE * SIZE)
        for (x in 0 until SIZE) dct1(rows, x, SIZE, out, x, SIZE)
        return out
    }

    private fun dct1(src: DoubleArray, srcOff: Int, srcStride: Int, dst: DoubleArray, dstOff: Int, dstStride: Int) {
        for (k in 0 until SIZE) {
            var acc = 0.0
            for (n in 0 until SIZE) {
                acc += src[srcOff + n * srcStride] * COS[k * SIZE + n]
            }
            dst[dstOff + k * dstStride] = acc
        }
    }

    private const val SIZE = 32
    private const val MID = 256
    private const val LOW = 8
    private val COS = DoubleArray(SIZE * SIZE) { i ->
        val k = i / SIZE
        val n = i % SIZE
        cos(PI * (2 * n + 1) * k / (2.0 * SIZE))
    }
}
