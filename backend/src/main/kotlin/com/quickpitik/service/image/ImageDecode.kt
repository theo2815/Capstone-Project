package com.quickpitik.service.image

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.IOException
import javax.imageio.ImageIO

object ImageDecode {
    fun readOrNull(bytes: ByteArray): BufferedImage? = try {
        ImageIO.read(ByteArrayInputStream(bytes))
    } catch (_: IOException) {
        null
    }
}
