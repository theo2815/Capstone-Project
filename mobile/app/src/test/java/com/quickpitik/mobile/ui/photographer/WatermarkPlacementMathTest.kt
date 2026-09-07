package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.remote.WatermarkSettingsDto
import org.junit.Assert.assertEquals
import org.junit.Test

// The preview must land the logo where the backend bakes it. These lock the
// free-space formula shared with the backend + website.
class WatermarkPlacementMathTest {

    @Test
    fun `default 1,1 lands bottom-right with the edge pad`() {
        // 3:1 logo, 15% scale, on a 400x300 box.
        val b = watermarkBox(400f, 300f, 3f, 0.15f, 1f, 1f)
        val pad = 400f * 0.025f // 10
        assertEquals(60f, b.width, 0.01f) // 400 * 0.15
        assertEquals(20f, b.height, 0.01f) // 60 / 3
        assertEquals(400f - 60f - pad, b.left, 0.01f) // 330
        assertEquals(300f - 20f - pad, b.top, 0.01f) // 270
    }

    @Test
    fun `0,0 lands top-left with the edge pad`() {
        val b = watermarkBox(400f, 300f, 3f, 0.15f, 0f, 0f)
        assertEquals(10f, b.left, 0.01f)
        assertEquals(10f, b.top, 0.01f)
    }

    @Test
    fun `aspect ratio is preserved at any scale`() {
        val b = watermarkBox(400f, 300f, 2f, 0.25f, 0.5f, 0.5f)
        assertEquals(2f, b.width / b.height, 0.001f)
    }

    @Test
    fun `positionFromBox is the inverse of watermarkBox`() {
        val b = watermarkBox(400f, 300f, 3f, 0.2f, 0.3f, 0.7f)
        val (x, y) = positionFromBox(400f, 300f, 3f, 0.2f, b.left, b.top)
        assertEquals(0.3f, x, 0.001f)
        assertEquals(0.7f, y, 0.001f)
    }

    @Test
    fun `positions clamp into the frame`() {
        val b = watermarkBox(400f, 300f, 3f, 0.15f, 5f, -2f)
        assertEquals(400f - 60f - 10f, b.left, 0.01f) // clamped to bottom-right
        assertEquals(10f, b.top, 0.01f) // clamped to top
    }

    @Test
    fun `pos and withPos target only the chosen orientation`() {
        val moved = WatermarkSettingsDto().withPos(landscape = false, x = 0.2, y = 0.8)
        assertEquals(0.2 to 0.8, moved.pos(landscape = false))
        assertEquals(1.0 to 1.0, moved.pos(landscape = true)) // landscape untouched
    }
}
