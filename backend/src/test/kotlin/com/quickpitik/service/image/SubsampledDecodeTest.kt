package com.quickpitik.service.image

import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO
import kotlin.test.assertEquals

class SubsampledDecodeTest {

    @Test
    fun `chooses the coarsest subsampling that leaves the target long edge`() {
        assertEquals(3, SubsampledDecode.subsampleFactor(5000, 3000, 1280))
        assertEquals(1, SubsampledDecode.subsampleFactor(1200, 800, 1280))
        assertEquals(2, SubsampledDecode.subsampleFactor(6000, 4000, 2048))
    }

    @Test
    fun `reads a subsampled raster`() {
        val decoded = SubsampledDecode.readForLongEdge(jpeg(3000, 2000), 1280)

        assertEquals(1500, maxOf(decoded!!.width, decoded.height))
    }

    private fun jpeg(width: Int, height: Int): ByteArray {
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        val graphics = image.createGraphics()
        try {
            graphics.color = Color.BLUE
            graphics.fillRect(0, 0, width, height)
        } finally {
            graphics.dispose()
        }
        return ByteArrayOutputStream().also { ImageIO.write(image, "jpeg", it) }.toByteArray()
    }
}
