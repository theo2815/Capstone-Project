package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.remote.WatermarkSettingsDto
import org.junit.Assert.assertEquals
import org.junit.Test

class WatermarkPlacementMathTest {

    @Test
    fun `render-pixel geometry matches the backend and website fixtures`() {
        assertEquals(WatermarkBox(328, 268, 60, 20), watermarkBox(400, 300, 3.0, 0.15, 1.0, 1.0))
        assertEquals(WatermarkBox(20, 20, 48, 16), watermarkBox(800, 1200, 3.0, 0.05, 0.0, 0.0))
        assertEquals(WatermarkBox(97, 144, 52, 17), watermarkBox(333, 222, 3.0, 0.157, 0.33, 0.73))
        assertEquals(WatermarkBox(854, 32, 394, 789), watermarkBox(1280, 853, 0.5, 0.4, 1.0, 1.0))
        assertEquals(WatermarkBox(30, 9, 1, 2), watermarkBox(40, 20, 0.5, 0.4, 1.0, 1.0))
    }

    @Test
    fun `rendered position round-trips through the drag inverse including fitted logos`() {
        for (aspect in listOf(0.5, 3.0, 100.0)) {
            val box = watermarkBox(1280, 853, aspect, 0.4, 0.3, 0.7)
            val (x, y) = positionFromBox(1280, 853, aspect, 0.4, box.left.toDouble(), box.top.toDouble(), 0.3, 0.7)
            assertEquals(box, watermarkBox(1280, 853, aspect, 0.4, x, y))
        }
    }

    @Test
    fun `drag clamps at edges and remains finite when the logo fills the available height`() {
        assertEquals(0.0 to 0.7, positionFromBox(1280, 853, 0.5, 0.4, -100.0, -100.0, 1.0, 0.7))
        assertEquals(1.0 to 0.7, positionFromBox(1280, 853, 0.5, 0.4, 2000.0, 2000.0, 1.0, 0.7))
        assertEquals(WatermarkBox(328, 12, 60, 20), watermarkBox(400, 300, 3.0, 0.15, 5.0, -2.0))
    }

    @Test
    fun `pos and withPos target only the chosen orientation`() {
        val moved = WatermarkSettingsDto().withPos(landscape = false, x = 0.2, y = 0.8)
        assertEquals(0.2 to 0.8, moved.pos(landscape = false))
        assertEquals(1.0 to 1.0, moved.pos(landscape = true))
    }
}
