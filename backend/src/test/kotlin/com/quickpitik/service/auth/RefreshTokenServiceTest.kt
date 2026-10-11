package com.quickpitik.service.auth

import com.quickpitik.config.JwtProperties
import com.quickpitik.entity.RefreshToken
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.repository.RefreshTokenRepository
import com.quickpitik.security.OpaqueTokens
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import java.time.Duration
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

// Every caller in the suite mocks this service, so rotation itself — the
// property that a leaked refresh token dies the moment it is used — was never
// exercised, and neither was the change-password carve-out that keeps the
// device that changed the password signed in (2026-09-11 test-suite audit).
class RefreshTokenServiceTest {

    private lateinit var repository: RefreshTokenRepository
    private val userId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()


    @BeforeEach
    fun setUp() {
        repository = Mockito.mock(RefreshTokenRepository::class.java)
        Mockito.`when`(repository.save(anyArg<RefreshToken>())).thenAnswer { it.arguments[0] as RefreshToken }
    }

    private fun service() = RefreshTokenService(
        repository,
        JwtProperties(
            secret = "test-secret-min-32-bytes-for-HS256-signing-purposes-only-do-not-use-in-prod",
            accessTokenTtl = Duration.ofMinutes(15),
            refreshTokenTtl = Duration.ofDays(7),
            refreshRotationGrace = Duration.ofMinutes(2),
        ),
    )

    private fun stored(
        plaintext: String,
        expiresAt: OffsetDateTime = OffsetDateTime.now().plusDays(1),
        revokedAt: OffsetDateTime? = null,
    ): RefreshToken {
        val token = RefreshToken(
            userId = userId,
            tokenHash = OpaqueTokens.hash(plaintext),
            expiresAt = expiresAt,
            revokedAt = revokedAt,
        )
        Mockito.`when`(repository.findByTokenHash(token.tokenHash)).thenReturn(token)
        return token
    }

    @Test
    fun `rotating revokes the parent and issues a different token for the same user`() {
        val parent = stored("parent-token")

        val (owner, child) = service().validateAndRotate("parent-token")

        assertEquals(userId, owner)
        assertNotEquals("parent-token", child)
        assertNotNull(parent.revokedAt, "the presented token must be dead after rotation")
        val saved = ArgumentCaptor.forClass(RefreshToken::class.java)
        Mockito.verify(repository, Mockito.times(2)).save(saved.capture())
        val issued = saved.allValues.last()
        assertEquals(OpaqueTokens.hash(child), issued.tokenHash)
        assertEquals(userId, issued.userId)
        assertTrue(issued.isActive())
    }

    @Test
    fun `a revoked or expired token is refused and nothing new is issued`() {
        stored("revoked-token", revokedAt = OffsetDateTime.now().minusMinutes(1))
        stored("expired-token", expiresAt = OffsetDateTime.now().minusMinutes(1))

        for (presented in listOf("revoked-token", "expired-token")) {
            val ex = assertFailsWith<UnauthorizedException> { service().validateAndRotate(presented) }
            assertEquals("INVALID_REFRESH_TOKEN", ex.code, presented)
        }
        Mockito.verify(repository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `an unknown token is refused`() {
        val ex = assertFailsWith<UnauthorizedException> { service().validateAndRotate("never-issued") }

        assertEquals("INVALID_REFRESH_TOKEN", ex.code)
        Mockito.verify(repository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `rotation records which child replaced the parent`() {
        val parent = stored("parent-token")

        val (_, child) = service().validateAndRotate("parent-token")

        val saved = ArgumentCaptor.forClass(RefreshToken::class.java)
        Mockito.verify(repository, Mockito.times(2)).save(saved.capture())
        val issued = saved.allValues.first { it.tokenHash == OpaqueTokens.hash(child) }
        assertEquals(issued.id, parent.replacedById)
    }

    // A rotation whose answer was lost: the client still holds the parent and
    // presents it again. Inside the grace window it gets a fresh pair and the
    // child it never received dies, so one token stays live, not two.
    @Test
    fun `a parent presented again inside the grace window is rotated once more and its orphan child dies`() {
        val orphan = stored("orphan-child")
        val parent = stored("parent-token", revokedAt = OffsetDateTime.now().minusSeconds(30)).apply { replacedById = orphan.id }
        Mockito.`when`(repository.findById(orphan.id)).thenReturn(Optional.of(orphan))

        val (owner, child) = service().validateAndRotate("parent-token")

        assertEquals(userId, owner)
        assertNotEquals("orphan-child", child)
        assertNotNull(orphan.revokedAt, "the child nobody received must be dead")
        assertNotEquals(orphan.id, parent.replacedById)
        val saved = ArgumentCaptor.forClass(RefreshToken::class.java)
        Mockito.verify(repository, Mockito.atLeast(3)).save(saved.capture())
        assertTrue(saved.allValues.any { it.tokenHash == OpaqueTokens.hash(child) && it.isActive() })
        assertEquals(saved.allValues.last { it.tokenHash == parent.tokenHash }.replacedById, saved.allValues.first { it.tokenHash == OpaqueTokens.hash(child) }.id)
    }

    @Test
    fun `outside the grace window, or revoked by a sign-out, the parent stays dead`() {
        val orphan = stored("orphan-child")
        stored("late-parent", revokedAt = OffsetDateTime.now().minusMinutes(3)).apply { replacedById = orphan.id }
        stored("signed-out", revokedAt = OffsetDateTime.now().minusSeconds(5)) // revoked by logout: no child, no grace

        for (presented in listOf("late-parent", "signed-out")) {
            val ex = assertFailsWith<UnauthorizedException> { service().validateAndRotate(presented) }
            assertEquals("INVALID_REFRESH_TOKEN", ex.code, presented)
        }
        Mockito.verify(repository, Mockito.never()).save(anyArg())
    }

    // Arguments are captured through thenAnswer, not eq()/captors: Mockito's
    // matchers hand Kotlin a null that the call site null-checks before the
    // mock is ever reached (the same reason anyArg() exists).
    @Test
    fun `revokeAllForUserExcept keeps only the named device`() {
        var kept: List<Any?> = emptyList()
        Mockito.`when`(repository.revokeAllForUserExcept(anyArg(), anyArg(), anyArg()))
            .thenAnswer { kept = it.arguments.toList(); 0 }

        service().revokeAllForUserExcept(userId, "this-device")

        assertEquals(listOf(userId, OpaqueTokens.hash("this-device")), kept.take(2))
        Mockito.verify(repository, Mockito.never()).revokeAllForUser(anyArg(), anyArg())
    }

    @Test
    fun `revokeAllForUserExcept with nothing to keep revokes everything`() {
        var revokedFor: UUID? = null
        Mockito.`when`(repository.revokeAllForUser(anyArg(), anyArg()))
            .thenAnswer { revokedFor = it.getArgument(0); 0 }

        service().revokeAllForUserExcept(userId, " ")

        assertEquals(userId, revokedFor)
        Mockito.verify(repository, Mockito.never()).revokeAllForUserExcept(anyArg(), anyArg(), anyArg())
    }
}
