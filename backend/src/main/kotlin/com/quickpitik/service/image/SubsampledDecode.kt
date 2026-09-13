package com.quickpitik.service.image

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

object SubsampledDecode {
    fun subsampleFactor(width: Int, height: Int, targetLongEdge: Int): Int =
        (maxOf(width, height) / targetLongEdge).coerceAtLeast(1)

    fun readForLongEdge(bytes: ByteArray, targetLongEdge: Int): BufferedImage? =
        ImageIO.createImageInputStream(ByteArrayInputStream(bytes))?.use { stream ->
            val readers = ImageIO.getImageReaders(stream)
            if (!readers.hasNext()) return@use null
            val reader = readers.next()
            try {
                reader.input = stream
                val factor = subsampleFactor(reader.getWidth(0), reader.getHeight(0), targetLongEdge)
                reader.read(0, reader.defaultReadParam.apply { setSourceSubsampling(factor, factor, 0, 0) })
            } finally {
                reader.dispose()
            }
        }
}
