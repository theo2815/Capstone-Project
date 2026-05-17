package com.quickpitik.websocket

import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import org.springframework.web.socket.CloseStatus
import org.springframework.web.socket.WebSocketSession
import org.springframework.web.socket.handler.TextWebSocketHandler

@Component
class AdminNotificationWebSocketHandler(
    private val registry: AdminNotificationSessionRegistry,
) : TextWebSocketHandler() {
    private val log = LoggerFactory.getLogger(javaClass)

    override fun afterConnectionEstablished(session: WebSocketSession) {
        registry.add(session)
        log.debug("WebSocket connected (admin) session={}", session.id)
    }

    override fun afterConnectionClosed(session: WebSocketSession, status: CloseStatus) {
        registry.remove(session)
        log.debug("WebSocket closed (admin) session={} status={}", session.id, status)
    }
}
