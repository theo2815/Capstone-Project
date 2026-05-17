package com.quickpitik.websocket

import com.quickpitik.security.JwtTokenProvider
import io.jsonwebtoken.JwtException
import org.slf4j.LoggerFactory
import org.springframework.http.server.ServerHttpRequest
import org.springframework.http.server.ServerHttpResponse
import org.springframework.http.server.ServletServerHttpRequest
import org.springframework.stereotype.Component
import org.springframework.web.socket.WebSocketHandler
import org.springframework.web.socket.server.HandshakeInterceptor
import java.util.UUID

// JWT-only handshake for /ws/me/photographer/notifications. The role gate
// (PHOTOGRAPHER vs other roles) is enforced at use site — non-photographers
// can technically connect but their userId has no admin-decision rows
// flowing in, so the channel is silent. Keeping the gate loose here means
// runner/admin users who somehow point at this URL get a no-op instead of
// a confusing handshake failure. The data path is the gate.
@Component
class MeNotificationHandshakeInterceptor(
    private val jwtTokenProvider: JwtTokenProvider,
) : HandshakeInterceptor {
    private val log = LoggerFactory.getLogger(javaClass)

    override fun beforeHandshake(
        request: ServerHttpRequest,
        response: ServerHttpResponse,
        wsHandler: WebSocketHandler,
        attributes: MutableMap<String, Any>,
    ): Boolean {
        val token = extractToken(request)
        if (token == null) {
            log.debug("WS handshake rejected (me): missing token")
            return false
        }
        val userId = try {
            UUID.fromString(jwtTokenProvider.parse(token).subject)
        } catch (ex: JwtException) {
            log.debug("WS handshake rejected (me): invalid token ({})", ex.message)
            return false
        } catch (ex: IllegalArgumentException) {
            log.debug("WS handshake rejected (me): malformed subject ({})", ex.message)
            return false
        }
        attributes[MeNotificationWebSocketHandler.ATTR_USER_ID] = userId
        // Echo the chosen subprotocol back to the client. Browsers require
        // this when the client offered a Sec-WebSocket-Protocol value;
        // omitting the response header makes the upgrade malformed and
        // the browser drops the connection with code 1006 immediately
        // after the handshake.
        response.headers.set("Sec-WebSocket-Protocol", token)
        return true
    }

    override fun afterHandshake(
        request: ServerHttpRequest,
        response: ServerHttpResponse,
        wsHandler: WebSocketHandler,
        exception: Exception?,
    ) {
        // no-op
    }

    private fun extractToken(request: ServerHttpRequest): String? {
        val protocolHeader = request.headers.getFirst("Sec-WebSocket-Protocol")
        if (!protocolHeader.isNullOrBlank()) {
            return protocolHeader.split(",").map { it.trim() }.firstOrNull { it.isNotEmpty() }
        }
        if (request is ServletServerHttpRequest) {
            val servletRequest = request.servletRequest
            servletRequest.getParameter("token")?.takeIf { it.isNotBlank() }?.let { return it }
        }
        return null
    }
}
