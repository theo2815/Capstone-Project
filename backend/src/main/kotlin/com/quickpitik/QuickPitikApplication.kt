package com.quickpitik

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.context.properties.ConfigurationPropertiesScan
import org.springframework.boot.runApplication
import javax.imageio.ImageIO

@SpringBootApplication
@ConfigurationPropertiesScan
class QuickPitikApplication

fun main(args: Array<String>) {
    // Decode/encode through heap buffers, not java.io.tmpdir scratch files.
    ImageIO.setUseCache(false)
    runApplication<QuickPitikApplication>(*args)
}
