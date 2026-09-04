package com.quickpitik.auth

import com.quickpitik.config.JwtProperties
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.security.JwtTokenProvider
import io.jsonwebtoken.JwtException
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.time.Duration
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class JwtTokenProviderTest {
    private val props = JwtProperties(
        secret = "test-secret-min-32-bytes-for-HS256-signing-purposes-only-do-not-use-in-prod",
        accessTokenTtl = Duration.ofMinutes(15),
        refreshTokenTtl = Duration.ofDays(7),
    )
    private val provider = JwtTokenProvider(props)

    @Test
    fun `signed token round-trips with expected claims`() {
        val user = newUser()

        val token = provider.createAccessToken(user)
        val claims = provider.parse(token)

        assertEquals(user.id.toString(), claims.subject)
        assertEquals("test@example.com", claims["email"])
        assertEquals("RUNNER", claims["role"])
        assertEquals(false, claims["suspended"])
        assertNotNull(claims.issuedAt)
        assertNotNull(claims.expiration)
        assertTrue(claims.expiration.after(claims.issuedAt))
    }

    // Backstop claim: both mint paths refuse a suspended user today, so this
    // only fires if a future mint path forgets. JwtAuthenticationFilter reads it.
    @Test
    fun `suspended user's token carries the suspended claim`() {
        val user = newUser().apply { suspendedAt = OffsetDateTime.now() }

        val claims = provider.parse(provider.createAccessToken(user))

        assertEquals(true, claims["suspended"])
    }

    @Test
    fun `tampered token is rejected`() {
        val user = newUser()
        val token = provider.createAccessToken(user)
        val tampered = token.dropLast(4) + "xxxx"

        assertThrows<JwtException> { provider.parse(tampered) }
    }

    private fun newUser(): User = User(
        id = UUID.randomUUID(),
        email = "test@example.com",
        passwordHash = "ignored",
        name = "Test",
        role = Role.RUNNER,
    )
}
