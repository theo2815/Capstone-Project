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
import java.time.Duration
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

// Spotted 2026-08-14: a signed-out visitor's upgrade was refused outright
// (close 1006, no onopen), so guests never got live photos. Opened up on the
// product call that this channel carries nothing the permitAll event grid
// doesn't already serve. An invalid token must still fail, or an expired
// session hides behind a socket that looks healthy.
class EventPhotoHandshakeInterceptorTest {

    private val jwtTokenProvider = JwtTokenProvider(
        JwtProperties(
            secret = "test-secret-min-32-bytes-for-HS256-signing-purposes-only-do-not-use-in-prod",
            accessTokenTtl = Duration.ofMinutes(15),
            refreshTokenTtl = Duration.ofDays(7),
        ),
    )
    private val interceptor = EventPhotoHandshakeInterceptor(jwtTokenProvider)
    private val handler = Mockito.mock(WebSocketHandler::class.java)

    private val eventId = UUID.randomUUID()

    @Test
    fun `a guest with no token is admitted and lands in the right event`() {
        val attributes = mutableMapOf<String, Any>()

        assertTrue(shake(attributes).admitted)
        assertEquals(eventId, attributes[EventPhotoWebSocketHandler.ATTR_EVENT_ID])
    }

    @Test
    fun `a guest handshake echoes no subprotocol`() {
        // The browser offers none, and selecting one it never sent is the same
        // malformed upgrade that produced the original 1006.
        assertNull(shake().protocolEcho)
    }

    @Test
    fun `a guest session carries no user id`() {
        val attributes = mutableMapOf<String, Any>()

        shake(attributes)

        assertNull(attributes[EventPhotoWebSocketHandler.ATTR_USER_ID])
    }

    @Test
    fun `a signed-in subscriber still gets the user id and the protocol echo`() {
        val user = user()
        val token = jwtTokenProvider.createAccessToken(user)
        val attributes = mutableMapOf<String, Any>()

        val result = shake(attributes, token)

        assertTrue(result.admitted)
        assertEquals(user.id, attributes[EventPhotoWebSocketHandler.ATTR_USER_ID])
        assertEquals(token, result.protocolEcho)
    }

    @Test
    fun `a present but invalid token is still refused`() {
        val tampered = jwtTokenProvider.createAccessToken(user()).dropLast(4) + "xxxx"

        assertFalse(shake(token = tampered).admitted)
    }

    @Test
    fun `a non-UUID event segment is still refused`() {
        assertFalse(shake(path = "/ws/events/not-a-uuid/photos").admitted)
    }

    private data class Handshake(val admitted: Boolean, val protocolEcho: String?)

    private fun shake(
        attributes: MutableMap<String, Any> = mutableMapOf(),
        token: String? = null,
        path: String = "/ws/events/$eventId/photos",
    ): Handshake {
        val request = MockHttpServletRequest("GET", path)
        token?.let { request.addHeader("Sec-WebSocket-Protocol", it) }
        val response = ServletServerHttpResponse(MockHttpServletResponse())
        val admitted = interceptor.beforeHandshake(
            ServletServerHttpRequest(request),
            response,
            handler,
            attributes,
        )
        return Handshake(admitted, response.headers.getFirst("Sec-WebSocket-Protocol"))
    }

    private fun user(): User = User(
        id = UUID.randomUUID(),
        email = "runner@test.local",
        passwordHash = "ignored",
        name = "Test Runner",
        role = Role.RUNNER,
    )
}
