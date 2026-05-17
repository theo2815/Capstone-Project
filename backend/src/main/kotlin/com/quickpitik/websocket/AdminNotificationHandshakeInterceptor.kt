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

// JWT + ADMIN-role handshake for /ws/admin/notifications. Non-admins are
// rejected at handshake so they never see broadcasts. Role is read from
// the JWT "role" claim (set by JwtTokenProvider.createAccessToken).
@Component
class AdminNotificationHandshakeInterceptor(
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
            log.debug("WS handshake rejected (admin): missing token")
            return false
        }
        val claims = try {
            jwtTokenProvider.parse(token)
        } catch (ex: JwtException) {
            log.debug("WS handshake rejected (admin): invalid token ({})", ex.message)
            return false
        }
        val role = claims["role"] as? String
        if (role != "ADMIN") {
            log.debug("WS handshake rejected (admin): non-admin role={}", role)
            return false
        }
        // Echo the chosen subprotocol back to the client (the JWT itself
        // — see MeNotificationHandshakeInterceptor for the rationale).
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
