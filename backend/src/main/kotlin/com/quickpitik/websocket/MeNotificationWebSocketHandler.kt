package com.quickpitik.websocket

import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import org.springframework.web.socket.CloseStatus
import org.springframework.web.socket.WebSocketSession
import org.springframework.web.socket.handler.TextWebSocketHandler
import java.util.UUID

@Component
class MeNotificationWebSocketHandler(
    private val registry: MeNotificationSessionRegistry,
) : TextWebSocketHandler() {
    private val log = LoggerFactory.getLogger(javaClass)

    override fun afterConnectionEstablished(session: WebSocketSession) {
        val userId = session.attributes[ATTR_USER_ID] as? UUID
        if (userId == null) {
            session.close(CloseStatus.POLICY_VIOLATION.withReason("missing userId"))
            return
        }
        registry.add(userId, session)
        log.debug("WebSocket connected (me) userId={} session={}", userId, session.id)
    }

    override fun afterConnectionClosed(session: WebSocketSession, status: CloseStatus) {
        val userId = session.attributes[ATTR_USER_ID] as? UUID ?: return
        registry.remove(userId, session)
        log.debug("WebSocket closed (me) userId={} session={} status={}", userId, session.id, status)
    }

    companion object {
        const val ATTR_USER_ID = "meUserId"
    }
}
