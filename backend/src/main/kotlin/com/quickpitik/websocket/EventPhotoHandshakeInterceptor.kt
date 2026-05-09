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

@Component
class EventPhotoHandshakeInterceptor(
    private val jwtTokenProvider: JwtTokenProvider,
) : HandshakeInterceptor {
    private val log = LoggerFactory.getLogger(javaClass)

    override fun beforeHandshake(
        request: ServerHttpRequest,
        response: ServerHttpResponse,
        wsHandler: WebSocketHandler,
        attributes: MutableMap<String, Any>,
    ): Boolean {
        val eventId = extractEventId(request)
        if (eventId == null) {
            log.debug("WS handshake rejected: bad eventId in {}", request.uri.path)
            return false
        }
        val token = extractToken(request)
        if (token == null) {
            log.debug("WS handshake rejected: missing token for event {}", eventId)
            return false
        }
        val userId = try {
            UUID.fromString(jwtTokenProvider.parse(token).subject)
        } catch (ex: JwtException) {
            log.debug("WS handshake rejected: invalid token for event {} ({})", eventId, ex.message)
            return false
        } catch (ex: IllegalArgumentException) {
            log.debug("WS handshake rejected: malformed subject for event {} ({})", eventId, ex.message)
            return false
        }
        attributes[EventPhotoWebSocketHandler.ATTR_EVENT_ID] = eventId
        attributes[EventPhotoWebSocketHandler.ATTR_USER_ID] = userId
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

    private fun extractEventId(request: ServerHttpRequest): UUID? {
        val segments = request.uri.path.trimEnd('/').split("/")
        val idx = segments.indexOf("events")
        if (idx == -1 || idx + 1 >= segments.size) return null
        return runCatching { UUID.fromString(segments[idx + 1]) }.getOrNull()
    }

    private fun extractToken(request: ServerHttpRequest): String? {
        val protocolHeader = request.headers.getFirst("Sec-WebSocket-Protocol")
        if (!protocolHeader.isNullOrBlank()) {
            // FE sends comma-separated protocols; we accept the first non-empty token
            return protocolHeader.split(",").map { it.trim() }.firstOrNull { it.isNotEmpty() }
        }
        if (request is ServletServerHttpRequest) {
            val servletRequest = request.servletRequest
            servletRequest.getParameter("token")?.takeIf { it.isNotBlank() }?.let { return it }
        }
        return null
    }
}
