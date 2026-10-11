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
import kotlin.test.assertNull
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
    // presents it again. Inside the grace window it gets one more pair. The
    // first child is left alone: a second browser tab may have received it
    // (two tabs refreshing at once look the same as a lost answer), and
    // killing it would sign both tabs out.
    @Test
    fun `a parent presented again inside the grace window gets one more token and its first child stays live`() {
        val firstChild = stored("first-child")
        val parent = stored("parent-token", revokedAt = OffsetDateTime.now().minusSeconds(30)).apply { replacedById = firstChild.id }
        Mockito.`when`(repository.findById(firstChild.id)).thenReturn(Optional.of(firstChild))

        val (owner, child) = service().validateAndRotate("parent-token")

        assertEquals(userId, owner)
        assertNotEquals("first-child", child)
        assertNull(firstChild.revokedAt, "the first child may be in another tab's hands")
        assertEquals(firstChild.id, parent.replacedById, "grace stays tied to the first child")
        val saved = ArgumentCaptor.forClass(RefreshToken::class.java)
        Mockito.verify(repository, Mockito.times(1)).save(saved.capture())
        assertEquals(OpaqueTokens.hash(child), saved.value.tokenHash)
        assertTrue(saved.value.isActive())
    }

    @Test
    fun `outside the grace window, or once its child is revoked, or revoked by a sign-out, the parent stays dead`() {
        val orphan = stored("orphan-child")
        Mockito.`when`(repository.findById(orphan.id)).thenReturn(Optional.of(orphan))
        stored("late-parent", revokedAt = OffsetDateTime.now().minusMinutes(3)).apply { replacedById = orphan.id }
        stored("signed-out", revokedAt = OffsetDateTime.now().minusSeconds(5)) // revoked by logout: no child, no grace
        // A password change or "revoke all" kills the child; the parent's grace dies with it.
        val deadChild = stored("dead-child", revokedAt = OffsetDateTime.now().minusSeconds(5))
        stored("family-revoked", revokedAt = OffsetDateTime.now().minusSeconds(30)).apply { replacedById = deadChild.id }
        Mockito.`when`(repository.findById(deadChild.id)).thenReturn(Optional.of(deadChild))

        for (presented in listOf("late-parent", "signed-out", "family-revoked")) {
            val ex = assertFailsWith<UnauthorizedException> { service().validateAndRotate(presented) }
            assertEquals("INVALID_REFRESH_TOKEN", ex.code, presented)
        }
        Mockito.verify(repository, Mockito.never()).save(anyArg())
    }

    // A client whose rotation answer was lost still holds the parent, so its
    // sign-out presents the parent. That must end the grace, not restart it.
    @Test
    fun `signing out with an already-rotated token ends its grace`() {
        val child = stored("first-child")
        Mockito.`when`(repository.findById(child.id)).thenReturn(Optional.of(child))
        stored("parent-token", revokedAt = OffsetDateTime.now().minusMinutes(3)).apply { replacedById = child.id }

        service().revoke("parent-token")

        assertFailsWith<UnauthorizedException> { service().validateAndRotate("parent-token") }
    }

    // Binds the shipped application.yml so the production default is what is
    // tested. Ten minutes: an upload from a venue with a poor connection must
    // survive a lost rotation answer followed by several minutes offline.
    @Test
    fun `the shipped grace window is ten minutes`() {
        val environment = org.springframework.mock.env.MockEnvironment()
        org.springframework.boot.env.YamlPropertySourceLoader()
            .load("application", org.springframework.core.io.ClassPathResource("application.yml"))
            .forEach { environment.propertySources.addLast(it) }
        val shipped = RefreshTokenService(
            repository,
            org.springframework.boot.context.properties.bind.Binder.get(environment)
                .bindOrCreate("jwt", JwtProperties::class.java),
        )
        val child = stored("first-child")
        Mockito.`when`(repository.findById(child.id)).thenReturn(Optional.of(child))
        stored("nine-minutes", revokedAt = OffsetDateTime.now().minusMinutes(9)).apply { replacedById = child.id }
        stored("eleven-minutes", revokedAt = OffsetDateTime.now().minusMinutes(11)).apply { replacedById = child.id }

        shipped.validateAndRotate("nine-minutes")

        assertFailsWith<UnauthorizedException> { shipped.validateAndRotate("eleven-minutes") }
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

    // The kept device's token stays live, so without this its just-rotated
    // parent would keep its grace through the password change.
    @Test
    fun `revokeAllForUserExcept ends the grace of the kept token's parent`() {
        val kept = stored("this-device")
        val parent = stored("parent-token", revokedAt = OffsetDateTime.now().minusSeconds(30)).apply { replacedById = kept.id }
        Mockito.`when`(repository.findById(kept.id)).thenReturn(Optional.of(kept))
        Mockito.`when`(repository.findByReplacedById(kept.id)).thenReturn(parent)

        service().revokeAllForUserExcept(userId, "this-device")

        assertFailsWith<UnauthorizedException> { service().validateAndRotate("parent-token") }
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
