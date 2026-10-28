package com.quickpitik.mobile.data

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import kotlin.math.roundToInt

// Face search only needs the face: a selfie leaves the phone at most 1280 px
// on its long edge, JPEG quality 85, pixels already upright. Same rule as the
// website's shrinkSelfie in website/src/lib/image-utils.ts.
internal const val SELFIE_MAX_EDGE = 1280
private const val SELFIE_JPEG_QUALITY = 85

/** The size that fits [maxEdge] with the aspect kept; never larger than the original. */
internal fun targetSize(w: Int, h: Int, maxEdge: Int = SELFIE_MAX_EDGE): Pair<Int, Int> {
    val longEdge = maxOf(w, h)
    if (longEdge <= maxEdge) return w to h
    val scale = maxEdge.toDouble() / longEdge
    return maxOf(1, (w * scale).roundToInt()) to maxOf(1, (h * scale).roundToInt())
}

/** The largest power-of-two decode step that still leaves the long edge at or above [maxEdge]. */
internal fun sampleSize(w: Int, h: Int, maxEdge: Int = SELFIE_MAX_EDGE): Int {
    var size = 1
    while (maxOf(w, h) / (size * 2) >= maxEdge) size *= 2
    return size
}

/** The transform that turns a photo's stored pixels upright; null when they already are. */
internal fun exifMatrix(orientation: Int): Matrix? {
    val matrix = Matrix()
    when (orientation) {
        ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
        ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
        ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
        ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
        ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
        else -> return null
    }
    return matrix
}

/**
 * The selfie as an upright JPEG no larger than [SELFIE_MAX_EDGE]. Bytes this
 * phone can't decode (or run out of memory on) go back unchanged, exactly as
 * they were sent before, and the backend judges them.
 */
internal fun shrinkSelfie(bytes: ByteArray): ByteArray = runCatching shrink@{
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
    // No MIME type is Android's "not an image I can decode".
    if (bounds.outMimeType == null || bounds.outWidth <= 0 || bounds.outHeight <= 0) return@shrink bytes
    val decoded = BitmapFactory.decodeByteArray(
        bytes, 0, bytes.size,
        BitmapFactory.Options().apply { inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight) },
    ) ?: return@shrink bytes
    val orientation = runCatching {
        ExifInterface(ByteArrayInputStream(bytes))
            .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
    }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)
    val (w, h) = targetSize(decoded.width, decoded.height)
    val matrix = (exifMatrix(orientation) ?: Matrix())
        .apply { preScale(w / decoded.width.toFloat(), h / decoded.height.toFloat()) }
    val upright = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
    ByteArrayOutputStream().use {
        upright.compress(Bitmap.CompressFormat.JPEG, SELFIE_JPEG_QUALITY, it)
        it.toByteArray()
    }
}.getOrDefault(bytes)
