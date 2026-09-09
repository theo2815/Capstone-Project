package com.quickpitik.mobile.ui.common

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.IntRect
import androidx.compose.ui.unit.IntSize
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AvatarCropGeometryTest {
    private val viewport = IntSize(256, 256)

    @Test
    fun portraitAndLandscapePanCanReachTheImageEdgesWithoutShrinkingTheCrop() {
        assertEquals(IntRect(0, 0, 400, 400), avatarCropRect(IntSize(800, 400), viewport, 1f, Offset(10000f, 10000f)))
        assertEquals(IntRect(400, 0, 800, 400), avatarCropRect(IntSize(800, 400), viewport, 1f, Offset(-10000f, -10000f)))
        assertEquals(IntRect(0, 0, 400, 400), avatarCropRect(IntSize(400, 800), viewport, 1f, Offset(10000f, 10000f)))
        assertEquals(IntRect(0, 400, 400, 800), avatarCropRect(IntSize(400, 800), viewport, 1f, Offset(-10000f, -10000f)))
    }

    @Test
    fun zoomingOutReclampsThePreviewAndExportToTheSameSquare() {
        for (image in listOf(IntSize(800, 400), IntSize(400, 800), IntSize(511, 511))) {
            val zoomedPan = clampAvatarOffset(image, viewport, 5f, Offset(-10000f, 10000f))
            for (scale in listOf(5f, 2.5f, 1f)) {
                val pan = clampAvatarOffset(image, viewport, scale, zoomedPan)
                val preview = avatarCropRect(image, viewport, scale, pan)
                assertEquals(preview, avatarCropRect(image, viewport, scale, zoomedPan))
                assertEquals(preview.width, preview.height)
                assertTrue(preview.width > 0 && preview.left >= 0 && preview.top >= 0)
                assertTrue(preview.right <= image.width && preview.bottom <= image.height)
            }
        }
        assertEquals(Offset.Zero, clampAvatarOffset(IntSize(511, 511), viewport, 1f, Offset(10000f, 10000f)))
    }
}
