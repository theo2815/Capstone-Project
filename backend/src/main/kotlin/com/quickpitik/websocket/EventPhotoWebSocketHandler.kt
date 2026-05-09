package com.quickpitik.websocket

import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import org.springframework.web.socket.CloseStatus
import org.springframework.web.socket.WebSocketSession
import org.springframework.web.socket.handler.TextWebSocketHandler
import java.util.UUID

@Component
class EventPhotoWebSocketHandler(
    private val registry: EventPhotoSessionRegistry,
) : TextWebSocketHandler() {
    private val log = LoggerFactory.getLogger(javaClass)

    override fun afterConnectionEstablished(session: WebSocketSession) {
        val eventId = session.attributes[ATTR_EVENT_ID] as? UUID
        if (eventId == null) {
            session.close(CloseStatus.POLICY_VIOLATION.withReason("missing eventId"))
            return
        }
        registry.add(eventId, session)
        log.debug("WebSocket connected eventId={} session={}", eventId, session.id)
    }

    override fun afterConnectionClosed(session: WebSocketSession, status: CloseStatus) {
        val eventId = session.attributes[ATTR_EVENT_ID] as? UUID ?: return
        registry.remove(eventId, session)
        log.debug("WebSocket closed eventId={} session={} status={}", eventId, session.id, status)
    }

    companion object {
        const val ATTR_EVENT_ID = "eventId"
        const val ATTR_USER_ID = "userId"
    }
}
