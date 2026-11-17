package com.quickpitik.service.liveness

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IdentityProperties
import com.quickpitik.entity.LivenessPurpose
import com.quickpitik.entity.LivenessSession
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.repository.LivenessSessionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.ai.RekognitionIdentityClient
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.mockito.Mockito.mock
import org.mockito.Mockito.never
import org.mockito.Mockito.verify
import org.mockito.Mockito.`when`
import org.springframework.beans.factory.ObjectProvider
import org.springframework.http.HttpStatus
import org.springframework.transaction.TransactionDefinition
import org.springframework.transaction.support.AbstractPlatformTransactionManager
import org.springframework.transaction.support.DefaultTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import software.amazon.awssdk.core.exception.SdkClientException
import software.amazon.awssdk.services.sts.model.Credentials
import java.time.Clock
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull

class LivenessSessionServiceTest {
    private object NoopTxManager : AbstractPlatformTransactionManager() {
        override fun doGetTransaction(): Any = Any()
        override fun doBegin(transaction: Any, definition: TransactionDefinition) = Unit
        override fun doCommit(status: DefaultTransactionStatus) = Unit
        override fun doRollback(status: DefaultTransactionStatus) = Unit
    }

    private lateinit var sessions: LivenessSessionRepository
    private lateinit var users: UserRepository
    private lateinit var client: RekognitionIdentityClient
    private lateinit var provider: ObjectProvider<RekognitionIdentityClient>
    private val userId = UUID.randomUUID()
    private val roleArn = "arn:aws:iam::000000000000:role/quickpitik-liveness-browser"

    // 2026-11-09 23:59 in Manila (UTC+8).
    private val lateEvening = Clock.fixed(Instant.parse("2026-11-09T15:59:00Z"), ZoneOffset.UTC)

    private fun <T> anyArg(): T = Mockito.any()

    @Suppress("UNCHECKED_CAST")
    @BeforeEach
    fun setUp() {
        sessions = mock(LivenessSessionRepository::class.java)
        users = mock(UserRepository::class.java)
        client = mock(RekognitionIdentityClient::class.java)
        provider = mock(ObjectProvider::class.java) as ObjectProvider<RekognitionIdentityClient>
        `when`(provider.getIfAvailable()).thenReturn(client)
        `when`(client.createLivenessSession()).thenReturn("aws-session-1")
        `when`(client.browserCredentials("aws-session-1")).thenReturn(
            Credentials.builder().accessKeyId("AKIA-TEST").secretAccessKey("secret").sessionToken("token")
                .expiration(Instant.parse("2026-11-09T16:14:00Z")).build(),
        )
    }

    private fun service(
        clock: Clock = lateEvening,
        aiOn: Boolean = true,
        identity: IdentityProperties = IdentityProperties(livenessRoleArn = roleArn),
    ) = LivenessSessionService(
        sessions, users, provider, AiApiProperties(enabled = aiOn), identity, TransactionTemplate(NoopTxManager), clock,
    )

    private fun countSoFar(n: Long) =
        `when`(sessions.countByUserIdAndCreatedAtGreaterThanEqual(anyArg(), anyArg())).thenReturn(n)

    private fun startPhotographer(svc: LivenessSessionService = service()) =
        svc.start(userId, LivenessPurpose.PHOTOGRAPHER_IDENTITY, null, 3, "1.2")

    @Test
    fun `fourth session in one Manila day is refused without calling AWS`() {
        countSoFar(3)
        val ex = assertFailsWith<ApiException> { startPhotographer() }
        assertEquals(HttpStatus.TOO_MANY_REQUESTS, ex.status)
        assertEquals(ErrorCodes.LIVENESS_ATTEMPTS_EXHAUSTED, ex.code)
        // 23:59 Manila: the next Manila midnight is one minute away.
        assertEquals(60L, ex.retryAfterSeconds)
        verify(client, never()).createLivenessSession()
        verify(sessions, never()).save(anyArg())
    }

    @Test
    fun `the 429 carries Retry-After and no retryAfterSeconds in the body`() {
        countSoFar(3)
        val ex = assertFailsWith<ApiException> { startPhotographer() }
        val response = GlobalExceptionHandler().handleApi(ex)
        assertEquals("60", response.headers.getFirst("Retry-After"))
        val body = ObjectMapper().writeValueAsString(response.body)
        assertFalse("retryAfterSeconds" in body, body)
    }

    @Test
    fun `ai off answers liveness unavailable`() {
        val ex = assertFailsWith<ApiException> { startPhotographer(service(aiOn = false)) }
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, ex.status)
        assertEquals(ErrorCodes.LIVENESS_UNAVAILABLE, ex.code)
        verify(client, never()).createLivenessSession()
    }

    @Test
    fun `no rekognition provider or no browser role answers liveness unavailable`() {
        `when`(provider.getIfAvailable()).thenReturn(null)
        assertEquals(ErrorCodes.LIVENESS_UNAVAILABLE, assertFailsWith<ApiException> { startPhotographer() }.code)

        `when`(provider.getIfAvailable()).thenReturn(client)
        val noRole = service(identity = IdentityProperties(livenessRoleArn = " "))
        assertEquals(ErrorCodes.LIVENESS_UNAVAILABLE, assertFailsWith<ApiException> { startPhotographer(noRole) }.code)
        verify(client, never()).createLivenessSession()
    }

    @Test
    fun `aws failure maps to liveness unavailable with no provider text`() {
        countSoFar(0)
        `when`(client.createLivenessSession()).thenThrow(SdkClientException.create("Unable to reach rekognition.ap-southeast-5"))
        val ex = assertFailsWith<ApiException> { startPhotographer() }
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, ex.status)
        assertEquals(ErrorCodes.LIVENESS_UNAVAILABLE, ex.code)
        assertFalse("rekognition" in ex.message!!.lowercase(), ex.message)
        verify(sessions, never()).save(anyArg())
    }

    @Test
    fun `a credentials failure saves no row so it does not use up an attempt`() {
        countSoFar(0)
        `when`(client.browserCredentials("aws-session-1")).thenThrow(SdkClientException.create("sts down"))
        assertEquals(ErrorCodes.LIVENESS_UNAVAILABLE, assertFailsWith<ApiException> { startPhotographer() }.code)
        verify(sessions, never()).save(anyArg())
    }

    @Test
    fun `the row is bound to the caller and counts from Manila midnight`() {
        countSoFar(1)
        val since = ArgumentCaptor.forClass(OffsetDateTime::class.java)

        val dto = startPhotographer(service(clock = lateEvening))
        // 00:01 Manila on 2026-11-10: a new day.
        startPhotographer(service(clock = Clock.fixed(Instant.parse("2026-11-09T16:01:00Z"), ZoneOffset.UTC)))

        verify(sessions, Mockito.atLeastOnce()).countByUserIdAndCreatedAtGreaterThanEqual(Mockito.eq(userId) ?: userId, since.capture() ?: OffsetDateTime.MIN)
        // Each start counts twice (before AWS and again under the lock); the first and last read show both days.
        assertEquals(Instant.parse("2026-11-08T16:00:00Z"), since.allValues.first().toInstant())
        assertEquals(Instant.parse("2026-11-09T16:00:00Z"), since.allValues.last().toInstant())

        assertEquals("aws-session-1", dto.sessionId)
        assertEquals("ap-southeast-5", dto.region)
        assertEquals(1, dto.attemptsLeftToday)
        assertEquals("AKIA-TEST", dto.credentials!!.accessKeyId)
        assertEquals(Instant.parse("2026-11-09T16:14:00Z"), dto.credentials!!.expiration)
        assertNull(dto.identityPoolId)
        verify(users, Mockito.atLeastOnce()).lockById(userId)
    }

    @Test
    fun `the row saves the purpose the event and the consent version`() {
        countSoFar(0)
        val eventId = UUID.randomUUID()
        val saved = ArgumentCaptor.forClass(LivenessSession::class.java)

        service().start(userId, LivenessPurpose.RUNNER_PHOTO_REMOVAL, eventId, 2, "1.3")

        verify(sessions).save(saved.capture() ?: LivenessSession("x", userId, LivenessPurpose.PHOTOGRAPHER_IDENTITY, createdAt = OffsetDateTime.MIN))
        val row = saved.value
        assertEquals("aws-session-1", row.sessionId)
        assertEquals(userId, row.userId)
        assertEquals(LivenessPurpose.RUNNER_PHOTO_REMOVAL, row.purpose)
        assertEquals(eventId, row.eventId)
        assertEquals("1.3", row.consentVersion)
        assertEquals(Instant.parse("2026-11-09T15:59:00Z"), row.createdAt.toInstant())
        assertNull(row.consumedAt)
    }

    @Test
    fun `the daily count spans every purpose`() {
        // Three photographer checks today already: a runner-removal start the same day is refused too.
        countSoFar(3)
        val ex = assertFailsWith<ApiException> {
            service().start(userId, LivenessPurpose.RUNNER_PHOTO_REMOVAL, UUID.randomUUID(), 3, "1.3")
        }
        assertEquals(ErrorCodes.LIVENESS_ATTEMPTS_EXHAUSTED, ex.code)
    }

    @Test
    fun `a session started for one purpose or event is not open for another`() {
        val eventA = UUID.randomUUID()
        val row = LivenessSession("s-1", userId, LivenessPurpose.RUNNER_PHOTO_REMOVAL, eventA, "1.3", OffsetDateTime.now(lateEvening))
        `when`(sessions.findBySessionIdAndUserIdAndPurpose("s-1", userId, LivenessPurpose.RUNNER_PHOTO_REMOVAL)).thenReturn(row)

        assertEquals(row, service().requireOpen(userId, "s-1", LivenessPurpose.RUNNER_PHOTO_REMOVAL, eventA))
        for ((purpose, event) in listOf(
            LivenessPurpose.RUNNER_PHOTO_REMOVAL to UUID.randomUUID(),
            LivenessPurpose.PHOTOGRAPHER_IDENTITY to null,
        )) {
            val ex = assertFailsWith<ApiException> { service().requireOpen(userId, "s-1", purpose, event) }
            assertEquals(HttpStatus.CONFLICT, ex.status)
            assertEquals(ErrorCodes.LIVENESS_SESSION_INVALID, ex.code)
        }

        row.consumedAt = OffsetDateTime.now(lateEvening)
        assertEquals(
            ErrorCodes.LIVENESS_SESSION_INVALID,
            assertFailsWith<ApiException> { service().requireOpen(userId, "s-1", LivenessPurpose.RUNNER_PHOTO_REMOVAL, eventA) }.code,
        )
    }
}
