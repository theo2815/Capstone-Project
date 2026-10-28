package com.quickpitik.mobile.data

import android.media.ExifInterface
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * R1: a selfie leaves the phone at ≤ 1280 px, upright, JPEG — or exactly as
 * picked when Android can't decode it. Robolectric 4.11 has no real bitmap
 * codec on Windows, so the decode/encode itself is checked on a device.
 */
@RunWith(RobolectricTestRunner::class)
class SelfieShrinkTest {

    @Test fun `target size keeps the aspect and never upscales`() {
        assertEquals(1280 to 853, targetSize(6000, 4000))
        assertEquals(853 to 1280, targetSize(4000, 6000))
        assertEquals(800 to 600, targetSize(800, 600))
        assertEquals(1280 to 1280, targetSize(1280, 1280))
    }

    @Test fun `sample size halves only while the long edge stays at or above 1280`() {
        assertEquals(1, sampleSize(800, 600))
        assertEquals(1, sampleSize(2559, 1000))
        assertEquals(2, sampleSize(2560, 1000))
        assertEquals(4, sampleSize(8160, 6120))
    }

    @Test fun `a sideways portrait is turned upright, an upright one is left alone`() {
        val point = floatArrayOf(1f, 0f)
        exifMatrix(ExifInterface.ORIENTATION_ROTATE_90)!!.mapPoints(point)
        assertEquals(0f, point[0], 1e-6f)
        assertEquals(1f, point[1], 1e-6f)
        assertNull(exifMatrix(ExifInterface.ORIENTATION_NORMAL))
    }

    @Test fun `bytes Android cannot decode are sent unchanged`() {
        val junk = ByteArray(16) { 7 }
        assertArrayEquals(junk, shrinkSelfie(junk))
    }
}
