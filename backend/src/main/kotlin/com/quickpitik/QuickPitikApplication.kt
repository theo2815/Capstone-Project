package com.quickpitik

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.context.properties.ConfigurationPropertiesScan
import org.springframework.boot.runApplication

@SpringBootApplication
@ConfigurationPropertiesScan
class QuickPitikApplication

fun main(args: Array<String>) {
    runApplication<QuickPitikApplication>(*args)
}
