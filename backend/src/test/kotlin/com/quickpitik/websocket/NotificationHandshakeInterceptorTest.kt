package com.quickpitik.websocket

import com.quickpitik.config.JwtProperties
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.security.JwtTokenProvider
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.server.ServletServerHttpRequest
import org.springframework.http.server.ServletServerHttpResponse
import org.springframework.mock.web.MockHttpServletRequest
import org.springframework.mock.web.MockHttpServletResponse
import org.springframework.web.socket.WebSocketHandler
import org.springframework.web.socket.server.HandshakeInterceptor
import java.time.Duration
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// The event channel's handshake has six tests; the two notification channels
// — which DO require a token, and one of which is the admin inbox — had none
// (2026-09-11 test-suite audit). Pinned here: a token is mandatory on both,
// the admin channel refuses every non-ADMIN claim set, and neither channel
// accepts first-factor, step-up, or suspended access tokens.
class NotificationHandshakeInterceptorTest {

    private val jwtTokenProvider = JwtTokenProvider(
        JwtProperties(
            secret = "test-secret-min-32-bytes-for-HS256-signing-purposes-only-do-not-use-in-prod",
            accessTokenTtl = Duration.ofMinutes(15),
            refreshTokenTtl = Duration.ofDays(7),
        ),
    )
    private val me = MeNotificationHandshakeInterceptor(jwtTokenProvider)
    private val admin = AdminNotificationHandshakeInterceptor(jwtTokenProvider)
    private val handler = Mockito.mock(WebSocketHandler::class.java)

    private val mePath = "/ws/me/runner/notifications"
    private val adminPath = "/ws/admin/notifications"

    @Test
    fun `the me channel refuses a handshake with no token`() {
        assertFalse(shake(me, mePath).admitted)
    }

    @Test
    fun `the me channel refuses a tampered token`() {
        val tampered = token(Role.RUNNER).dropLast(4) + "xxxx"

        assertFalse(shake(me, mePath, tampered).admitted)
    }

    @Test
    fun `the me channel admits a signed-in user and echoes the protocol`() {
        val user = user(Role.RUNNER)
        val token = jwtTokenProvider.createAccessToken(user)
        val attributes = mutableMapOf<String, Any>()

        val result = shake(me, mePath, token, attributes)

        assertTrue(result.admitted)
        assertEquals(user.id, attributes[MeNotificationWebSocketHandler.ATTR_USER_ID])
        assertEquals(token, result.protocolEcho)
    }

    @Test
    fun `the me channel refuses an MFA first-factor token`() {
        assertFalse(shake(me, mePath, jwtTokenProvider.createMfaPendingToken(UUID.randomUUID())).admitted)
    }

    @Test
    fun `the me channel refuses a step-up token`() {
        assertFalse(shake(me, mePath, jwtTokenProvider.createStepUpToken(UUID.randomUUID())).admitted)
    }

    @Test
    fun `both notification channels refuse suspended access tokens`() {
        val runner = user(Role.RUNNER).apply { suspendedAt = OffsetDateTime.now() }
        val adminUser = user(Role.ADMIN).apply { suspendedAt = OffsetDateTime.now() }

        assertFalse(shake(me, mePath, jwtTokenProvider.createAccessToken(runner)).admitted)
        assertFalse(shake(admin, adminPath, jwtTokenProvider.createAccessToken(adminUser)).admitted)
    }

    @Test
    fun `the admin channel refuses a runner token`() {
        assertFalse(shake(admin, adminPath, token(Role.RUNNER)).admitted)
    }

    @Test
    fun `the admin channel refuses an MFA first-factor token`() {
        // Password accepted, TOTP not yet entered: the token has a valid
        // signature and subject but no role claim, so it must not open the
        // admin inbox.
        assertFalse(shake(admin, adminPath, jwtTokenProvider.createMfaPendingToken(UUID.randomUUID())).admitted)
    }

    @Test
    fun `the admin channel admits an admin and echoes the protocol`() {
        val token = token(Role.ADMIN)

        val result = shake(admin, adminPath, token)

        assertTrue(result.admitted)
        assertEquals(token, result.protocolEcho)
    }

    private data class Handshake(val admitted: Boolean, val protocolEcho: String?)

    private fun shake(
        interceptor: HandshakeInterceptor,
        path: String,
        token: String? = null,
        attributes: MutableMap<String, Any> = mutableMapOf(),
    ): Handshake {
        val request = MockHttpServletRequest("GET", path)
        token?.let { request.addHeader("Sec-WebSocket-Protocol", it) }
        val response = ServletServerHttpResponse(MockHttpServletResponse())
        val admitted = interceptor.beforeHandshake(ServletServerHttpRequest(request), response, handler, attributes)
        return Handshake(admitted, response.headers.getFirst("Sec-WebSocket-Protocol"))
    }

    private fun token(role: Role): String = jwtTokenProvider.createAccessToken(user(role))

    private fun user(role: Role): User = User(
        id = UUID.randomUUID(),
        email = "ws@test.local",
        passwordHash = "ignored",
        name = "Socket",
        role = role,
    )
}
