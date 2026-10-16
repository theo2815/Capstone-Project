package com.quickpitik.config

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AiPropertiesTest {

    @Test
    fun `face and bib default to Rekognition because ai-api no longer serves them`() {
        assertEquals(AiProvider.REKOGNITION, AiProperties().provider)
        val yml = javaClass.getResource("/application.yml")!!.readText()
        assertTrue("provider: \${AI_PROVIDER:rekognition}" in yml)
    }
}
