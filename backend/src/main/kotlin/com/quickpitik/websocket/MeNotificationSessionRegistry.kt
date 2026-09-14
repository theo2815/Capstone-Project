package com.quickpitik.websocket

import com.fasterxml.jackson.databind.ObjectMapper
import org.springframework.stereotype.Component
import org.springframework.web.socket.TextMessage
import org.springframework.web.socket.WebSocketSession
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

// Per-user session registry for /ws/me/photographer/notifications. A single
// photographer can have multiple open sessions (web + mobile + a second tab);
// every session for the same userId receives the broadcast so the bell
// updates everywhere at once.
//
// Mirrors EventPhotoSessionRegistry — same dead-session sweep on send.
@Component
class MeNotificationSessionRegistry(
    private val objectMapper: ObjectMapper,
    private val sender: WebSocketSender,
) {
    private val sessions = ConcurrentHashMap<UUID, ConcurrentHashMap<String, WebSocketSession>>()

    fun add(userId: UUID, session: WebSocketSession) {
        val targets = sessions.computeIfAbsent(userId) { ConcurrentHashMap() }
        targets[session.id] = sender.decorate(session) { targets.remove(it.id, it) }
    }

    fun remove(userId: UUID, session: WebSocketSession) {
        sessions[userId]?.remove(session.id)
    }

    fun broadcast(userId: UUID, payload: Any) {
        val targets = sessions[userId] ?: return
        if (targets.isEmpty()) return
        val message = TextMessage(objectMapper.writeValueAsString(payload))
        targets.values.forEach { session ->
            sender.send(session, message) { targets.remove(it.id, it) }
        }
    }

    fun sessionCount(userId: UUID): Int = sessions[userId]?.size ?: 0
}
