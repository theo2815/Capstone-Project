package com.quickpitik.websocket

import com.fasterxml.jackson.databind.ObjectMapper
import org.slf4j.LoggerFactory
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
) {
    private val log = LoggerFactory.getLogger(javaClass)
    private val sessions: ConcurrentHashMap<UUID, MutableSet<WebSocketSession>> = ConcurrentHashMap()

    fun add(userId: UUID, session: WebSocketSession) {
        sessions.computeIfAbsent(userId) { ConcurrentHashMap.newKeySet() }.add(session)
    }

    fun remove(userId: UUID, session: WebSocketSession) {
        sessions[userId]?.remove(session)
    }

    fun broadcast(userId: UUID, payload: Any) {
        val targets = sessions[userId] ?: return
        if (targets.isEmpty()) return
        val message = TextMessage(objectMapper.writeValueAsString(payload))
        val dead = mutableListOf<WebSocketSession>()
        targets.forEach { session ->
            try {
                if (session.isOpen) {
                    session.sendMessage(message)
                } else {
                    dead.add(session)
                }
            } catch (ex: Exception) {
                log.warn("WebSocket send failed for user {}: {}", userId, ex.message)
                dead.add(session)
            }
        }
        targets.removeAll(dead.toSet())
    }

    fun sessionCount(userId: UUID): Int = sessions[userId]?.size ?: 0
}
