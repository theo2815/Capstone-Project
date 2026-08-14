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

/**
 * Handshake gate for `/ws/events/{id}/photos` — and that endpoint only
 * (`WebSocketConfig:26`), unlike [MeNotificationHandshakeInterceptor], which
 * two endpoints share.
 *
 * **A token is optional here.** The channel pushes `photo.published` frames
 * carrying id / bib / tone / span / watermarked-thumbnail URL / uploadedAt —
 * byte-for-byte what the permitAll `GET /events/{slug}/photos` already serves
 * to anyone, so a signed-out spectator learns nothing they couldn't get by
 * polling. Nothing downstream is per-user either: [EventPhotoWebSocketHandler]
 * and [EventPhotoSessionRegistry] key purely on eventId. Refusing the upgrade
 * only cost guests live updates (2026-08-14 product call).
 *
 * A token that is *present but invalid* still fails the handshake — silently
 * downgrading it to anonymous would hide expired-session bugs behind a socket
 * that looks healthy.
 */
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
        if (token != null) {
            val userId = try {
                UUID.fromString(jwtTokenProvider.parse(token).subject)
            } catch (ex: JwtException) {
                log.debug("WS handshake rejected: invalid token for event {} ({})", eventId, ex.message)
                return false
            } catch (ex: IllegalArgumentException) {
                log.debug("WS handshake rejected: malformed subject for event {} ({})", eventId, ex.message)
                return false
            }
            attributes[EventPhotoWebSocketHandler.ATTR_USER_ID] = userId
            // Echo the chosen subprotocol back to the client. Without this,
            // browsers drop the connection with code 1006 immediately after
            // the handshake because the upgrade response is malformed (the
            // server accepted the handshake but didn't select a subprotocol
            // from the client's offer). Bug since Q-002 — pre-existed this
            // notifications PR but caught here while debugging the same
            // failure mode on the admin/photographer channels.
            //
            // Only when the client actually offered one: echoing a protocol a
            // guest never sent is the same malformed upgrade in reverse.
            response.headers.set("Sec-WebSocket-Protocol", token)
        } else {
            log.debug("WS handshake accepted anonymously for event {}", eventId)
        }
        attributes[EventPhotoWebSocketHandler.ATTR_EVENT_ID] = eventId
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
