package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IdentityProperties
import com.quickpitik.entity.LivenessPurpose
import com.quickpitik.entity.LivenessSession
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SellerReasonCode
import com.quickpitik.exception.ApiException
import com.quickpitik.service.ai.FaceCompareResult
import com.quickpitik.service.ai.LivenessResult
import com.quickpitik.service.ai.RekognitionIdentityClient
import com.quickpitik.service.liveness.LivenessSessionService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.mockito.Mockito.mock
import org.mockito.Mockito.never
import org.mockito.Mockito.verify
import org.mockito.Mockito.`when`
import org.springframework.beans.factory.ObjectProvider
import org.springframework.http.HttpStatus
import software.amazon.awssdk.core.exception.SdkClientException
import software.amazon.awssdk.services.rekognition.model.LivenessSessionStatus
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class IdentityCheckServiceTest {
    private lateinit var sessions: LivenessSessionService
    private lateinit var client: RekognitionIdentityClient
    private lateinit var provider: ObjectProvider<RekognitionIdentityClient>
    private val userId = UUID.randomUUID()
    private val sessionId = "s-1"
    private val consentAt = OffsetDateTime.parse("2026-11-09T01:00:00Z")
    private val reference = byteArrayOf(1, 1, 1)
    private val idFront = byteArrayOf(2, 2, 2)
    private val purpose = LivenessPurpose.PHOTOGRAPHER_IDENTITY

    private fun <T> anyArg(): T = Mockito.any()

    @Suppress("UNCHECKED_CAST")
    @BeforeEach
    fun setUp() {
        sessions = mock(LivenessSessionService::class.java)
        client = mock(RekognitionIdentityClient::class.java)
        provider = mock(ObjectProvider::class.java) as ObjectProvider<RekognitionIdentityClient>
        `when`(provider.getIfAvailable()).thenReturn(client)
        `when`(sessions.requireOpen(userId, sessionId, purpose, null))
            .thenReturn(LivenessSession(sessionId, userId, purpose, null, "1.2", consentAt))
        `when`(sessions.claim(userId, sessionId, purpose, null)).thenReturn(true)
        liveness(LivenessSessionStatus.SUCCEEDED, 99.5f)
        compare(FaceCompareResult(facesOnTarget = 1, bestSimilarity = 99.8f))
    }

    private fun service(aiEnabled: Boolean = true) =
        IdentityCheckService(sessions, provider, AiApiProperties(enabled = aiEnabled), IdentityProperties())

    private fun liveness(status: LivenessSessionStatus, confidence: Float?, image: ByteArray? = reference) {
        `when`(client.livenessResult(sessionId)).thenReturn(LivenessResult(status, confidence, image))
    }

    private fun compare(result: FaceCompareResult) {
        `when`(client.compareFaces(anyArg(), anyArg())).thenReturn(result)
    }

    private fun run(session: String? = sessionId) = service().run(userId, session) { idFront }

    @Test
    fun `compare uses the reference image as source and the id front as target never the holding-id selfie`() {
        val checks = run()

        // The only compare: source = live reference, target = ID front. run() never sees the holding-ID selfie.
        verify(client).compareFaces(reference, idFront)
        verify(client, Mockito.times(1)).compareFaces(anyArg(), anyArg())
        assertEquals(emptyList(), checks.reasons)
        assertEquals(BigDecimal("99.50"), checks.livenessConfidence)
        assertEquals(BigDecimal("99.80"), checks.faceSimilarity)
        assertContentEquals(reference, checks.referenceImage)
        assertEquals(sessionId, checks.sessionId)
        assertEquals(consentAt, checks.consentAt)
    }

    @Test
    fun `no session is declined with no aws call`() {
        val checks = run(session = null)

        assertEquals(listOf(SellerReasonCode.DECLINED), checks.reasons)
        assertNull(checks.consentAt)
        Mockito.verifyNoInteractions(sessions, client)
    }

    @Test
    fun `ai off is ai disabled`() {
        val checks = service(aiEnabled = false).run(userId, sessionId) { idFront }

        assertEquals(listOf(SellerReasonCode.AI_DISABLED), checks.reasons)
        verify(sessions).claim(userId, sessionId, purpose, null) // already used up
        assertEquals(consentAt, checks.consentAt)
        Mockito.verifyNoInteractions(client)
    }

    @Test
    fun `no client bean is ai disabled`() {
        `when`(provider.getIfAvailable()).thenReturn(null)

        assertEquals(listOf(SellerReasonCode.AI_DISABLED), run().reasons)
    }

    @Test
    fun `low liveness and low similarity are both recorded`() {
        liveness(LivenessSessionStatus.SUCCEEDED, 70.25f)
        compare(FaceCompareResult(facesOnTarget = 1, bestSimilarity = 98.9f))

        val checks = run()

        assertEquals(listOf(SellerReasonCode.LIVENESS_LOW, SellerReasonCode.FACE_MISMATCH), checks.reasons)
        assertEquals(BigDecimal("70.25"), checks.livenessConfidence)
        assertEquals(BigDecimal("98.90"), checks.faceSimilarity)
    }

    @Test
    fun `no face on the id front`() {
        compare(FaceCompareResult(facesOnTarget = 0, bestSimilarity = null))

        val checks = run()

        assertEquals(listOf(SellerReasonCode.NO_FACE_ON_ID), checks.reasons)
        assertNull(checks.faceSimilarity)
    }

    @Test
    fun `best of several faces on the id is used`() {
        // A PH ID's small "ghost" photo is a second face; the client already returns the best match.
        compare(FaceCompareResult(facesOnTarget = 2, bestSimilarity = 99.6f))

        assertEquals(emptyList(), run().reasons)
    }

    @Test
    fun `aws error on compare gives ai error and keeps the liveness score`() {
        `when`(client.compareFaces(anyArg(), anyArg())).thenThrow(SdkClientException.create("boom"))

        val checks = run()

        assertEquals(listOf(SellerReasonCode.AI_ERROR), checks.reasons)
        assertEquals(BigDecimal("99.50"), checks.livenessConfidence)
        assertNull(checks.faceSimilarity)
    }

    @Test
    fun `aws error on the liveness result gives ai error`() {
        `when`(client.livenessResult(sessionId)).thenThrow(SdkClientException.create("boom"))

        assertEquals(listOf(SellerReasonCode.AI_ERROR), run().reasons)
        verify(client, never()).compareFaces(anyArg(), anyArg())
    }

    @Test
    fun `another account's session is refused before aws is called`() {
        `when`(sessions.requireOpen(userId, sessionId, purpose, null)).thenThrow(LivenessSessionService.invalidSession())

        val error = assertFailsWith<ApiException> { run() }

        assertEquals(ErrorCodes.LIVENESS_SESSION_INVALID, error.code)
        verify(sessions, never()).claim(anyArg(), anyArg(), anyArg(), Mockito.any())
        Mockito.verifyNoInteractions(client)
    }

    @Test
    fun `a session someone else claimed first is refused before aws is called`() {
        `when`(sessions.claim(userId, sessionId, purpose, null)).thenReturn(false)

        val error = assertFailsWith<ApiException> { run() }

        assertEquals(HttpStatus.CONFLICT, error.status)
        assertEquals(ErrorCodes.LIVENESS_SESSION_INVALID, error.code)
        Mockito.verifyNoInteractions(client)
    }

    @Test
    fun `an unfinished session answers session invalid and stays used`() {
        for (status in listOf(LivenessSessionStatus.CREATED, LivenessSessionStatus.IN_PROGRESS, LivenessSessionStatus.FAILED, LivenessSessionStatus.EXPIRED)) {
            liveness(status, null, image = null)

            val error = assertFailsWith<ApiException>(status.toString()) { run() }

            assertEquals(ErrorCodes.LIVENESS_SESSION_INVALID, error.code)
        }
        verify(client, never()).compareFaces(anyArg(), anyArg())
    }

    @Test
    fun `the session is claimed before the liveness result is read`() {
        run()

        val order = Mockito.inOrder(sessions, client)
        order.verify(sessions).claim(userId, sessionId, purpose, null)
        order.verify(client).livenessResult(sessionId)
    }

    @Test
    fun `status is auto approved only when nothing failed`() {
        assertEquals(SellerDetailsStatus.AUTO_APPROVED, IdentityCheckService.statusFor(emptyList()))
        assertEquals(SellerDetailsStatus.AUTO_APPROVED, IdentityCheckService.statusFor(listOf(SellerReasonCode.PASSED)))
        for (reason in SellerReasonCode.entries - SellerReasonCode.PASSED) {
            assertEquals(SellerDetailsStatus.MANUAL_REVIEW, IdentityCheckService.statusFor(listOf(reason)), reason.name)
        }
    }
}
