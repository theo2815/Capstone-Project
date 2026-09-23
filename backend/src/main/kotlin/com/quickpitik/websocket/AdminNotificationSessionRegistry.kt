package com.quickpitik.websocket

import com.fasterxml.jackson.databind.ObjectMapper
import org.springframework.stereotype.Component
import org.springframework.web.socket.TextMessage
import org.springframework.web.socket.WebSocketSession
import java.util.concurrent.ConcurrentHashMap

// Flat session map for /ws/admin/notifications — every connected admin
// receives every broadcast (verification submitted, dispute filed, payout
// report filed). No per-key routing; the audience is "all admins" and we
// only have one bootstrap admin in v1 anyway.
@Component
class AdminNotificationSessionRegistry(
    private val objectMapper: ObjectMapper,
    private val sender: WebSocketSender,
) {
    private val sessions = ConcurrentHashMap<String, WebSocketSession>()

    fun add(session: WebSocketSession) {
        sessions[session.id] = sender.decorate(session) { sessions.remove(it.id, it) }
    }

    fun remove(session: WebSocketSession) {
        sessions.remove(session.id)
    }

    fun broadcast(payload: Any) {
        if (sessions.isEmpty()) return
        val message = TextMessage(objectMapper.writeValueAsString(payload))
        sessions.values.forEach { session ->
            sender.send(session, message) { sessions.remove(it.id, it) }
        }
    }

    fun sessionCount(): Int = sessions.size
}
