package com.quickpitik.security

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.config.JwtProperties
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import io.jsonwebtoken.Jwts
import io.jsonwebtoken.security.Keys
import jakarta.servlet.FilterChain
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.mock.web.MockHttpServletRequest
import org.springframework.mock.web.MockHttpServletResponse
import org.springframework.security.core.context.SecurityContextHolder
import java.nio.charset.StandardCharsets
import java.time.Duration
import java.time.Instant
import java.time.OffsetDateTime
import java.util.Date
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// The `suspended` claim is a backstop — both mint paths already refuse a
// suspended user — but the filter has to honour it, and it has to keep honouring
// tokens minted before the claim existed.
class JwtAuthenticationFilterTest {

    private val secret = "test-secret-min-32-bytes-for-HS256-signing-purposes-only-do-not-use-in-prod"
    private val provider = JwtTokenProvider(
        JwtProperties(
            secret = secret,
            accessTokenTtl = Duration.ofMinutes(15),
            refreshTokenTtl = Duration.ofDays(7),
        ),
    )
    private val filter = JwtAuthenticationFilter(provider, ObjectMapper())

    @AfterEach
    fun clearContext() = SecurityContextHolder.clearContext()

    @Test
    fun `an active user's token authenticates`() {
        val user = newUser()

        runFilter(provider.createAccessToken(user))

        val principal = SecurityContextHolder.getContext().authentication?.principal as? AuthPrincipal
        assertNotNull(principal)
        assertEquals(user.id, principal.userId)
        assertEquals(Role.RUNNER, principal.role)
    }

    @Test
    fun `a suspended user's token leaves the context unauthenticated`() {
        val user = newUser().apply { suspendedAt = OffsetDateTime.now() }

        runFilter(provider.createAccessToken(user))

        assertNull(SecurityContextHolder.getContext().authentication)
    }

    // Anyone holding a token minted before this change has no `suspended` claim.
    // Absent must read as "active" or the fix would mass-log-out every session.
    @Test
    fun `a token minted before the suspended claim still authenticates`() {
        val user = newUser()

        runFilter(tokenWithoutSuspendedClaim(user))

        val principal = SecurityContextHolder.getContext().authentication?.principal as? AuthPrincipal
        assertNotNull(principal)
        assertEquals(user.id, principal.userId)
    }

    // An expired or malformed token must short-circuit with a 401 instead of
    // proceeding unauthenticated — on a public route like POST /orders the old
    // fallthrough silently minted guest orders for logged-in users.
    @Test
    fun `an expired token is rejected with a 401 before the chain runs`() {
        val request = MockHttpServletRequest()
        request.addHeader("Authorization", "Bearer ${expiredToken(newUser())}")
        val response = MockHttpServletResponse()
        val chain = Mockito.mock(FilterChain::class.java)

        filter.doFilter(request, response, chain)

        assertEquals(401, response.status)
        assertTrue(response.contentAsString.contains("UNAUTHORIZED"))
        Mockito.verifyNoInteractions(chain)
        assertNull(SecurityContextHolder.getContext().authentication)
    }

    @Test
    fun `a malformed token is rejected with a 401 before the chain runs`() {
        val request = MockHttpServletRequest()
        request.addHeader("Authorization", "Bearer not-a-jwt")
        val response = MockHttpServletResponse()
        val chain = Mockito.mock(FilterChain::class.java)

        filter.doFilter(request, response, chain)

        assertEquals(401, response.status)
        Mockito.verifyNoInteractions(chain)
    }

    // Anything the filter doesn't explicitly classify must fail loudly — a
    // swallowed exception would continue the chain unauthenticated, which on
    // public routes recreates the guest downgrade.
    @Test
    fun `an unexpected claim failure propagates instead of silently de-authenticating`() {
        val request = MockHttpServletRequest()
        request.addHeader("Authorization", "Bearer ${tokenWithoutEmailClaim(newUser())}")

        assertFailsWith<IllegalStateException> {
            filter.doFilter(request, MockHttpServletResponse(), Mockito.mock(FilterChain::class.java))
        }
    }

    private fun runFilter(token: String) {
        val request = MockHttpServletRequest()
        request.addHeader("Authorization", "Bearer $token")
        filter.doFilter(request, MockHttpServletResponse(), Mockito.mock(FilterChain::class.java))
    }

    private fun expiredToken(user: User): String {
        val past = Instant.now().minusSeconds(3600)
        return Jwts.builder()
            .subject(user.id.toString())
            .claim("email", user.email)
            .claim("role", user.role.name)
            .issuedAt(Date.from(past))
            .expiration(Date.from(past.plusSeconds(60)))
            .signWith(Keys.hmacShaKeyFor(secret.toByteArray(StandardCharsets.UTF_8)))
            .compact()
    }

    // Signed and unexpired, but missing the email claim the filter requires.
    private fun tokenWithoutEmailClaim(user: User): String {
        val now = Instant.now()
        return Jwts.builder()
            .subject(user.id.toString())
            .claim("role", user.role.name)
            .issuedAt(Date.from(now))
            .expiration(Date.from(now.plusSeconds(900)))
            .signWith(Keys.hmacShaKeyFor(secret.toByteArray(StandardCharsets.UTF_8)))
            .compact()
    }

    private fun tokenWithoutSuspendedClaim(user: User): String {
        val now = Instant.now()
        return Jwts.builder()
            .subject(user.id.toString())
            .claim("email", user.email)
            .claim("role", user.role.name)
            .issuedAt(Date.from(now))
            .expiration(Date.from(now.plusSeconds(900)))
            .signWith(Keys.hmacShaKeyFor(secret.toByteArray(StandardCharsets.UTF_8)))
            .compact()
    }

    private fun newUser(): User = User(
        id = UUID.randomUUID(),
        email = "runner@example.com",
        passwordHash = "ignored",
        name = "Test",
        role = Role.RUNNER,
    )
}
