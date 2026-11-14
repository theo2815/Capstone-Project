package com.quickpitik.service.relevance

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.service.ai.RekognitionAiClient
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class RelevanceSwitchTest {

    private val bothCutoffs = RelevanceProperties.Junk(blurCutoff = 40.0, darkCutoff = 20.0)

    private fun switch(
        mode: RelevanceMode,
        junk: RelevanceProperties.Junk = bothCutoffs,
        flagsEnabled: Boolean = true,
        aiEnabled: Boolean = true,
    ) = RelevanceSwitch(RelevanceProperties(mode = mode, junk = junk), AdminProperties(flagsEnabled = flagsEnabled),
        AiApiProperties(enabled = aiEnabled))

    @Test fun `shadow runs as record-only`() {
        val s = switch(RelevanceMode.SHADOW)
        assertEquals(RelevanceMode.RECORD_ONLY, s.mode)
        assertFalse(s.v1Flagging)
        assertEquals(RelevanceMode.RECORD_ONLY, switch(RelevanceMode.RECORD_ONLY).mode)
    }

    @Test fun `auto-hide with a cut-off unset runs as record-only`() {
        val noBlur = switch(RelevanceMode.AUTO_HIDE, junk = RelevanceProperties.Junk(darkCutoff = 20.0))
        assertEquals(RelevanceMode.RECORD_ONLY, noBlur.mode)
        assertFalse(noBlur.cutoffsSet)
        assertEquals(RelevanceMode.RECORD_ONLY,
            switch(RelevanceMode.AUTO_HIDE, junk = RelevanceProperties.Junk(blurCutoff = 40.0)).mode)
    }

    @Test fun `auto-hide with the flags switch off runs as record-only`() {
        assertEquals(RelevanceMode.RECORD_ONLY, switch(RelevanceMode.AUTO_HIDE, flagsEnabled = false).mode)
    }

    @Test fun `auto-hide with both cut-offs and the flags switch on is auto-hide`() {
        val s = switch(RelevanceMode.AUTO_HIDE)
        assertEquals(RelevanceMode.AUTO_HIDE, s.mode)
        assertTrue(s.cutoffsSet)
        assertFalse(s.v1Flagging)
    }

    @Test fun `off is off whatever else is set`() {
        val s = switch(RelevanceMode.OFF)
        assertEquals(RelevanceMode.OFF, s.mode)
        assertFalse(s.v1Flagging)
        assertTrue(s.cutoffsSet)
        assertTrue(s.aiEnabled)
    }

    @Test fun `flag keeps version 1 flagging and records only while the flags switch is on`() {
        val on = switch(RelevanceMode.FLAG)
        assertEquals(RelevanceMode.RECORD_ONLY, on.mode)
        assertTrue(on.v1Flagging)
        assertFalse(switch(RelevanceMode.FLAG, flagsEnabled = false).v1Flagging)
    }

    @Test fun `ai enabled follows the ai switch`() {
        assertFalse(switch(RelevanceMode.RECORD_ONLY, aiEnabled = false).aiEnabled)
    }

    @Test fun `measuring needs AI on and the Rekognition client`() {
        @Suppress("UNCHECKED_CAST")
        fun provider(client: RekognitionAiClient?) = (Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>)
            .also { Mockito.`when`(it.getIfAvailable()).thenReturn(client) }
        fun switch(aiEnabled: Boolean, client: RekognitionAiClient?) = RelevanceSwitch(RelevanceProperties(mode = RelevanceMode.RECORD_ONLY),
            AdminProperties(), AiApiProperties(enabled = aiEnabled), provider(client))
        val client = Mockito.mock(RekognitionAiClient::class.java)

        assertTrue(switch(aiEnabled = true, client = client).canMeasure)
        assertFalse(switch(aiEnabled = false, client = client).canMeasure)
        assertFalse(switch(aiEnabled = true, client = null).canMeasure, "the ai-api provider has no label check")
    }

    @Test fun `deciding needs measuring and a model key`() {
        @Suppress("UNCHECKED_CAST")
        val provider = (Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>)
            .also { Mockito.`when`(it.getIfAvailable()).thenReturn(Mockito.mock(RekognitionAiClient::class.java)) }
        fun switch(aiEnabled: Boolean, key: String) = RelevanceSwitch(RelevanceProperties(mode = RelevanceMode.RECORD_ONLY,
            llm = RelevanceProperties.Llm(apiKey = key)), AdminProperties(), AiApiProperties(enabled = aiEnabled), provider)

        assertTrue(switch(aiEnabled = true, key = "sk-ant-x").canDecide)
        assertFalse(switch(aiEnabled = true, key = "").canDecide, "no key, no call")
        assertFalse(switch(aiEnabled = false, key = "sk-ant-x").canDecide, "AI off gates every paid call")
    }
}
