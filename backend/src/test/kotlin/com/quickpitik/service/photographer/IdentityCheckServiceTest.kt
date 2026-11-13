package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IdentityProperties
import com.quickpitik.entity.LivenessPurpose
import com.quickpitik.entity.LivenessSession
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SellerIdType
import com.quickpitik.entity.SellerReasonCode
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ValidationException
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
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

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
        lines("REPUBLIC OF THE PHILIPPINES", nameLine, "N01-12-345678")
    }

    private val legalName = "Juan Dela Cruz"
    private val nameLine = "DELA CRUZ, JUAN ANDRES"

    // A set format for the default type: with none, a found number goes to an admin (R-1).
    private val withFormat = IdentityProperties(idNumberFormats = mapOf(SellerIdType.DRIVERS_LICENSE to "[A-Z0-9]{4,30}"))

    private fun service(aiEnabled: Boolean = true, props: IdentityProperties = withFormat) =
        IdentityCheckService(sessions, provider, AiApiProperties(enabled = aiEnabled), props)

    private fun lines(vararg text: String) {
        `when`(client.detectLines(anyArg())).thenReturn(text.toList())
    }

    private fun liveness(status: LivenessSessionStatus, confidence: Float?, image: ByteArray? = reference) {
        `when`(client.livenessResult(sessionId)).thenReturn(LivenessResult(status, confidence, image))
    }

    private fun compare(result: FaceCompareResult) {
        `when`(client.compareFaces(anyArg(), anyArg())).thenReturn(result)
    }

    private fun run(
        session: String? = sessionId, idNumber: String? = "N01-12-345678", idType: SellerIdType = SellerIdType.DRIVERS_LICENSE,
        legal: String = legalName,
    ) = service().run(userId, session, idType, idNumber, legal) { idFront }

    private fun hmac(idType: SellerIdType, normalised: String) = service().idNumberHmac(idType, normalised)

    @Test
    fun `typed number found on the id front passes`() {
        val checks = run(idNumber = "n01 12 345678")

        assertEquals(emptyList(), checks.reasons)
        verify(client).detectLines(idFront)
        assertEquals(hmac(SellerIdType.DRIVERS_LICENSE, "N0112345678"), checks.idNumberHmac)
    }

    @Test
    fun `a number not on the card is a mismatch`() {
        assertEquals(listOf(SellerReasonCode.ID_NUMBER_MISMATCH), run(idNumber = "N01-12-999999").reasons)
    }

    @Test
    fun `no text read is unreadable`() {
        lines()

        assertEquals(listOf(SellerReasonCode.ID_NUMBER_UNREADABLE), run().reasons)
    }

    @Test
    fun `a missing number is unreadable and stores no hash`() {
        for (missing in listOf(null, "  ")) {
            val checks = run(idNumber = missing)

            assertEquals(listOf(SellerReasonCode.ID_NUMBER_UNREADABLE), checks.reasons)
            assertNull(checks.idNumberHmac)
        }
        verify(client, never()).detectLines(anyArg())
    }

    @Test
    fun `the hash ignores case spaces and dashes and differs per id type`() {
        val a = run(idNumber = "n01-12 345678").idNumberHmac!!
        val b = run(idNumber = "N0112345678").idNumberHmac!!

        assertEquals(a, b)
        assertTrue(Regex("[0-9a-f]{64}").matches(a), a)
        assertNotEquals(a, hmac(SellerIdType.PASSPORT, "N0112345678"))
        assertNotEquals(a, IdentityCheckService(sessions, provider, AiApiProperties(), IdentityProperties(idNumberHmacSecret = "other"))
            .idNumberHmac(SellerIdType.DRIVERS_LICENSE, "N0112345678"))
    }

    @Test
    fun `the declined path skips DetectText and still stores the hash`() {
        val checks = run(session = null)

        assertEquals(listOf(SellerReasonCode.DECLINED), checks.reasons)
        assertEquals(hmac(SellerIdType.DRIVERS_LICENSE, "N0112345678"), checks.idNumberHmac)
        Mockito.verifyNoInteractions(sessions, client)
    }

    @Test
    fun `a partial number is a mismatch`() {
        lines(nameLine, "N01-12-345678")

        assertEquals(listOf(SellerReasonCode.ID_NUMBER_MISMATCH), run(idNumber = "12345678").reasons)
    }

    @Test
    fun `a number split across two lines is a mismatch`() {
        lines(nameLine, "N01-12", "345678")

        assertEquals(listOf(SellerReasonCode.ID_NUMBER_MISMATCH), run(idNumber = "N0112345678").reasons)
    }

    @Test
    fun `a number spaced into tokens on one line matches`() {
        lines(nameLine, "LICENSE NO. N01 12 345678 EXP")

        assertEquals(emptyList(), run(idNumber = "N0112345678").reasons)
    }

    @Test
    fun `a number that breaks the type's format is a validation error on idNumber`() {
        val strict = service(props = IdentityProperties(idNumberFormats = mapOf(SellerIdType.DRIVERS_LICENSE to "[A-Z]\\d{10}")))
        assertEquals(emptyList(), strict.run(userId, sessionId, SellerIdType.DRIVERS_LICENSE, "N01-12-345678", legalName) { idFront }.reasons)

        for (typed in listOf("N01-12-34567", "abc", "N0!12345678", "A".repeat(31))) {
            val error = assertFailsWith<ValidationException>(typed) {
                strict.run(userId, sessionId, SellerIdType.DRIVERS_LICENSE, typed, legalName) { idFront }
            }
            assertEquals("idNumber", error.field)
            assertFalse(typed in error.message.orEmpty())
        }
        verify(sessions, Mockito.times(1)).claim(userId, sessionId, purpose, null) // only the valid run used the session
    }

    @Test
    fun `a type with no format sends a matching number to an admin`() {
        val strict = service(props = IdentityProperties(idNumberFormats = mapOf(SellerIdType.PASSPORT to "P\\d{7}[A-Z]")))
        lines(nameLine, "1234")

        assertEquals(listOf(SellerReasonCode.ID_FORMAT_UNCHECKED), strict.run(userId, sessionId, SellerIdType.DRIVERS_LICENSE, "1234", legalName) { idFront }.reasons)
    }

    @Test
    fun `a word on the card typed as the id number goes to an admin when the type has no format`() {
        val checks = service(props = IdentityProperties()).run(userId, sessionId, SellerIdType.DRIVERS_LICENSE, "REPUBLIC", legalName) { idFront }

        assertEquals(listOf(SellerReasonCode.ID_FORMAT_UNCHECKED), checks.reasons)
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, IdentityCheckService.statusFor(checks.reasons))
    }

    @Test
    fun `a word on the card breaks a set format and is refused`() {
        val strict = service(props = IdentityProperties(idNumberFormats = mapOf(SellerIdType.DRIVERS_LICENSE to "[A-Z]\\d{10}")))

        val error = assertFailsWith<ValidationException> { strict.run(userId, sessionId, SellerIdType.DRIVERS_LICENSE, "REPUBLIC", legalName) { idFront } }

        assertEquals("idNumber", error.field)
        verify(sessions, never()).claim(anyArg(), anyArg(), anyArg(), Mockito.any())
    }

    @Test
    fun `the match list holds the hash of the number under every id type`() {
        val checks = run(idNumber = "n01-12 345678")

        assertEquals(SellerIdType.entries.map { hmac(it, "N0112345678") }, checks.idNumberMatchHmacs)
        assertEquals(7, checks.idNumberMatchHmacs.toSet().size)
        assertEquals(hmac(SellerIdType.DRIVERS_LICENSE, "N0112345678"), checks.idNumberHmac)
        assertEquals(checks.idNumberMatchHmacs, run(session = null).idNumberMatchHmacs) // the declined path matches too
        assertEquals(emptyList(), run(idNumber = null).idNumberMatchHmacs)
    }

    @Test
    fun `a legal name that is not on the id front goes to an admin`() {
        val checks = run(legal = "Maria Santos")

        assertEquals(listOf(SellerReasonCode.ID_NAME_MISMATCH), checks.reasons)
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, IdentityCheckService.statusFor(checks.reasons))
    }

    @Test
    fun `a legal name printed in another order with a middle name passes`() {
        assertEquals(emptyList(), run(legal = "Juan Andres Dela Cruz").reasons)
        assertEquals(emptyList(), run(legal = "juan dela cruz").reasons)
    }

    @Test
    fun `no text read adds no name reason`() {
        lines()

        assertEquals(listOf(SellerReasonCode.ID_NUMBER_UNREADABLE), run(legal = "Maria Santos").reasons)
    }

    @Test
    fun `an aws error on DetectText adds no name reason`() {
        `when`(client.detectLines(anyArg())).thenThrow(SdkClientException.create("boom"))

        assertEquals(listOf(SellerReasonCode.AI_ERROR), run(legal = "Maria Santos").reasons)
    }

    @Test
    fun `the admin-instead path makes no name check`() {
        assertEquals(listOf(SellerReasonCode.DECLINED), run(session = null, legal = "Maria Santos").reasons)
        Mockito.verifyNoInteractions(client)
    }

    @Test
    fun `ai off adds no name reason`() {
        val checks = service(aiEnabled = false).run(userId, sessionId, SellerIdType.DRIVERS_LICENSE, "N0112345678", "Maria Santos") { idFront }

        assertEquals(listOf(SellerReasonCode.AI_DISABLED), checks.reasons)
    }

    @Test
    fun `the id front text is read once for both checks`() {
        run(legal = "Maria Santos")

        verify(client, Mockito.times(1)).detectLines(idFront)
    }

    @Test
    fun `aws error on DetectText gives one ai error and keeps the hash`() {
        `when`(client.detectLines(anyArg())).thenThrow(SdkClientException.create("boom"))
        `when`(client.compareFaces(anyArg(), anyArg())).thenThrow(SdkClientException.create("boom"))

        val checks = run()

        assertEquals(listOf(SellerReasonCode.AI_ERROR), checks.reasons)
        assertEquals(hmac(SellerIdType.DRIVERS_LICENSE, "N0112345678"), checks.idNumberHmac)
    }

    @Test
    fun `ai off adds no id number reason and still stores the hash`() {
        val checks = service(aiEnabled = false).run(userId, sessionId, SellerIdType.DRIVERS_LICENSE, "N0112345678", legalName) { idFront }

        assertEquals(listOf(SellerReasonCode.AI_DISABLED), checks.reasons)
        assertEquals(hmac(SellerIdType.DRIVERS_LICENSE, "N0112345678"), checks.idNumberHmac)
    }

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
        val checks = service(aiEnabled = false).run(userId, sessionId, SellerIdType.DRIVERS_LICENSE, null, legalName) { idFront }

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
    fun `a finished session with no reference image gives ai error and no compare`() {
        liveness(LivenessSessionStatus.SUCCEEDED, 99.5f, image = null)

        val checks = run()

        assertEquals(listOf(SellerReasonCode.AI_ERROR), checks.reasons)
        assertNull(checks.referenceImage)
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
