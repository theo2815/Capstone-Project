package com.quickpitik.service.moderation

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.ModerationProperties
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.ModerationHit
import com.quickpitik.service.ai.RekognitionAiClient
import org.junit.jupiter.api.Test
import org.mockito.Mockito.mock
import org.mockito.Mockito.verifyNoInteractions
import org.mockito.Mockito.`when`
import org.springframework.beans.factory.ObjectProvider
import org.springframework.http.HttpStatus
import kotlin.test.assertEquals

class ImageModerationServiceTest {
    private val rekognition = mock(RekognitionAiClient::class.java)
    private val bytes = byteArrayOf(1, 2, 3)

    @Suppress("UNCHECKED_CAST")
    private fun service(aiEnabled: Boolean = true, client: RekognitionAiClient? = rekognition): ImageModerationService {
        val provider = mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>
        `when`(provider.getIfAvailable()).thenReturn(client)
        return ImageModerationService(AiApiProperties(enabled = aiEnabled), ModerationProperties(), provider)
    }

    private fun hits(vararg hits: ModerationHit) {
        `when`(rekognition.moderationLabels(bytes, 0.8)).thenReturn(hits.toList())
    }

    @Test
    fun `AI off returns UNCHECKED without calling AWS`() {
        assertEquals(ImageModerationResult(ImageModerationVerdict.UNCHECKED), service(aiEnabled = false).check(bytes))
        verifyNoInteractions(rekognition)
    }

    @Test
    fun `no Rekognition client returns UNCHECKED`() {
        assertEquals(ImageModerationVerdict.UNCHECKED, service(client = null).check(bytes).verdict)
    }

    @Test
    fun `a blocked parent label flags (v6 Explicit Nudity and v7 Explicit both)`() {
        hits(ModerationHit("Graphic Male Nudity", "Explicit Nudity", 0.97))
        assertEquals(
            ImageModerationResult(ImageModerationVerdict.FLAGGED, listOf("Graphic Male Nudity")),
            service().check(bytes),
        )
        hits(ModerationHit("Exposed Male Genitalia", "explicit", 0.95))
        assertEquals(
            ImageModerationResult(ImageModerationVerdict.FLAGGED, listOf("Exposed Male Genitalia")),
            service().check(bytes),
        )
    }

    @Test
    fun `Swimwear or Underwear stays CLEAN`() {
        hits(
            ModerationHit("Swimwear or Underwear", "Suggestive", 0.99),
            ModerationHit("Female Swimwear or Underwear", "Swimwear or Underwear", 0.98),
        )
        assertEquals(ImageModerationResult(ImageModerationVerdict.CLEAN), service().check(bytes))
    }

    @Test
    fun `an AWS error returns UNCHECKED`() {
        `when`(rekognition.moderationLabels(bytes, 0.8))
            .thenThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "AWS Rekognition detectModerationLabels failed"))
        assertEquals(ImageModerationVerdict.UNCHECKED, service().check(bytes).verdict)
    }
}
