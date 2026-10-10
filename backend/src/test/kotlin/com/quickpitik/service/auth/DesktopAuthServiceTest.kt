package com.quickpitik.service.auth

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.auth.AuthResponse
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.entity.DesktopAuthCode
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.DesktopAuthCodeRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.OpaqueTokens
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import java.time.Clock
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals

// The desktop connect hand-off. What matters: the code is never stored in the
// clear, a code works once, the verifier must match the challenge, and every
// code failure answers the same way so a caller learns nothing from the message.
class DesktopAuthServiceTest {

    private lateinit var codes: DesktopAuthCodeRepository
    private lateinit var users: UserRepository
    private lateinit var authService: AuthService
    private val clock: Clock = Clock.fixed(Instant.parse("2026-10-10T04:00:00Z"), ZoneOffset.UTC)
    private val now: OffsetDateTime get() = OffsetDateTime.now(clock)

    private val verifier = "v".repeat(43)
    private val challenge = DesktopAuthService.challengeOf(verifier)

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        codes = Mockito.mock(DesktopAuthCodeRepository::class.java)
        users = Mockito.mock(UserRepository::class.java)
        authService = Mockito.mock(AuthService::class.java)
    }

    private fun service() = DesktopAuthService(codes, users, authService, clock)

    private fun photographer(role: Role = Role.PHOTOGRAPHER) =
        User(email = "p@test.local", passwordHash = "x", name = "P", role = role)

    private fun stored(
        user: User,
        code: String,
        expiresAt: OffsetDateTime = now.plusMinutes(2),
        usedAt: OffsetDateTime? = null,
        storedChallenge: String = challenge,
    ) = DesktopAuthCode(
        userId = user.id,
        codeHash = OpaqueTokens.hash(code),
        codeChallenge = storedChallenge,
        expiresAt = expiresAt,
        usedAt = usedAt,
        createdAt = now.minusSeconds(5),
    )

    private fun pairFor(user: User) = AuthResponse(
        accessToken = "access",
        refreshToken = "refresh",
        user = UserDto(id = user.id, email = user.email, name = user.name, role = user.role, createdAt = now),
    )

    @Test
    fun `mint stores the hash of the code, never the code`() {
        val user = photographer()
        Mockito.`when`(users.findById(user.id)).thenReturn(Optional.of(user))
        Mockito.`when`(codes.save(anyArg())).thenAnswer { it.arguments[0] }

        val response = service().mint(user.id, challenge)

        val captor = ArgumentCaptor.forClass(DesktopAuthCode::class.java)
        Mockito.verify(codes).save(captor.capture())
        assertEquals(43, response.code.length)
        assertEquals(OpaqueTokens.hash(response.code), captor.value.codeHash)
        assertNotEquals(response.code, captor.value.codeHash)
        assertEquals(challenge, captor.value.codeChallenge)
        assertEquals(now.plusMinutes(2), captor.value.expiresAt)
        assertEquals(120L, response.expiresInSeconds)
    }

    @Test
    fun `mint refuses a malformed challenge before touching the database`() {
        assertFailsWith<ValidationException> { service().mint(UUID.randomUUID(), "too-short") }
        Mockito.verify(codes, Mockito.never()).save(anyArg())
    }

    @Test
    fun `mint refuses a suspended user`() {
        val user = photographer().apply { suspendedAt = now }
        Mockito.`when`(users.findById(user.id)).thenReturn(Optional.of(user))

        val ex = assertFailsWith<UnauthorizedException> { service().mint(user.id, challenge) }

        assertEquals(ErrorCodes.ACCOUNT_SUSPENDED, ex.code)
        Mockito.verify(codes, Mockito.never()).save(anyArg())
    }

    @Test
    fun `exchange redeems a live code once and mints the normal pair`() {
        val user = photographer()
        val code = OpaqueTokens.generate()
        val row = stored(user, code)
        Mockito.`when`(codes.findByCodeHash(OpaqueTokens.hash(code))).thenReturn(row)
        Mockito.`when`(codes.markUsed(row.id, now)).thenReturn(1)
        Mockito.`when`(users.findById(user.id)).thenReturn(Optional.of(user))
        val pair = pairFor(user)
        Mockito.`when`(authService.buildAuthResponse(user)).thenReturn(pair)

        assertEquals(pair, service().exchange(code, verifier))
        Mockito.verify(codes).markUsed(row.id, now)
    }

    @Test
    fun `exchange refuses an expired code`() {
        val user = photographer()
        val code = OpaqueTokens.generate()
        Mockito.`when`(codes.findByCodeHash(OpaqueTokens.hash(code)))
            .thenReturn(stored(user, code, expiresAt = now.minusSeconds(1)))

        val ex = assertFailsWith<UnauthorizedException> { service().exchange(code, verifier) }

        assertEquals(ErrorCodes.INVALID_TOKEN, ex.code)
        Mockito.verify(codes, Mockito.never()).markUsed(anyArg(), anyArg())
    }

    @Test
    fun `exchange refuses a code that was already used`() {
        val user = photographer()
        val code = OpaqueTokens.generate()
        Mockito.`when`(codes.findByCodeHash(OpaqueTokens.hash(code)))
            .thenReturn(stored(user, code, usedAt = now.minusSeconds(30)))

        val ex = assertFailsWith<UnauthorizedException> { service().exchange(code, verifier) }

        assertEquals(ErrorCodes.INVALID_TOKEN, ex.code)
    }

    @Test
    fun `exchange refuses the wrong verifier`() {
        val user = photographer()
        val code = OpaqueTokens.generate()
        Mockito.`when`(codes.findByCodeHash(OpaqueTokens.hash(code))).thenReturn(stored(user, code))

        val ex = assertFailsWith<UnauthorizedException> { service().exchange(code, "w".repeat(43)) }

        assertEquals(ErrorCodes.INVALID_TOKEN, ex.code)
        Mockito.verify(codes, Mockito.never()).markUsed(anyArg(), anyArg())
    }

    @Test
    fun `exchange refuses malformed inputs without a database lookup`() {
        assertFailsWith<UnauthorizedException> { service().exchange("abc", verifier) }
        assertFailsWith<UnauthorizedException> { service().exchange("c".repeat(43), "short") }
        Mockito.verify(codes, Mockito.never()).findByCodeHash(anyArg())
    }

    @Test
    fun `exchange refuses when another redeem won the race`() {
        val user = photographer()
        val code = OpaqueTokens.generate()
        val row = stored(user, code)
        Mockito.`when`(codes.findByCodeHash(OpaqueTokens.hash(code))).thenReturn(row)
        Mockito.`when`(codes.markUsed(row.id, now)).thenReturn(0)
        Mockito.`when`(users.findById(user.id)).thenReturn(Optional.of(user))

        val ex = assertFailsWith<UnauthorizedException> { service().exchange(code, verifier) }

        assertEquals(ErrorCodes.INVALID_TOKEN, ex.code)
        Mockito.verify(authService, Mockito.never()).buildAuthResponse(anyArg())
    }

    @Test
    fun `exchange refuses a suspended user with the suspension code`() {
        val user = photographer().apply { suspendedAt = now }
        val code = OpaqueTokens.generate()
        val row = stored(user, code)
        Mockito.`when`(codes.findByCodeHash(OpaqueTokens.hash(code))).thenReturn(row)
        Mockito.`when`(users.findById(user.id)).thenReturn(Optional.of(user))

        val ex = assertFailsWith<UnauthorizedException> { service().exchange(code, verifier) }

        assertEquals(ErrorCodes.ACCOUNT_SUSPENDED, ex.code)
        Mockito.verify(codes, Mockito.never()).markUsed(anyArg(), anyArg())
    }

    @Test
    fun `exchange refuses an account that is not a photographer`() {
        val user = photographer(role = Role.RUNNER)
        val code = OpaqueTokens.generate()
        val row = stored(user, code)
        Mockito.`when`(codes.findByCodeHash(OpaqueTokens.hash(code))).thenReturn(row)
        Mockito.`when`(users.findById(user.id)).thenReturn(Optional.of(user))

        val ex = assertFailsWith<UnauthorizedException> { service().exchange(code, verifier) }

        assertEquals(ErrorCodes.INVALID_TOKEN, ex.code)
        Mockito.verify(codes, Mockito.never()).markUsed(anyArg(), anyArg())
    }

    @Test
    fun `every code failure carries the same message`() {
        val user = photographer()
        val expired = OpaqueTokens.generate()
        val used = OpaqueTokens.generate()
        val live = OpaqueTokens.generate()
        Mockito.`when`(codes.findByCodeHash(OpaqueTokens.hash(expired)))
            .thenReturn(stored(user, expired, expiresAt = now.minusSeconds(1)))
        Mockito.`when`(codes.findByCodeHash(OpaqueTokens.hash(used)))
            .thenReturn(stored(user, used, usedAt = now))
        Mockito.`when`(codes.findByCodeHash(OpaqueTokens.hash(live))).thenReturn(stored(user, live))

        val messages = listOf(
            assertFailsWith<UnauthorizedException> { service().exchange(expired, verifier) }.message,
            assertFailsWith<UnauthorizedException> { service().exchange(used, verifier) }.message,
            assertFailsWith<UnauthorizedException> { service().exchange(live, "w".repeat(43)) }.message,
            assertFailsWith<UnauthorizedException> { service().exchange("z".repeat(43), verifier) }.message,
        )

        assertEquals(1, messages.toSet().size)
        assertEquals("This connect code is invalid or has expired.", messages.first())
    }
}
