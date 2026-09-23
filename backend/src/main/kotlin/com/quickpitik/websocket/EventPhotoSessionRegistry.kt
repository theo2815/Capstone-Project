package com.quickpitik.websocket

import com.fasterxml.jackson.databind.ObjectMapper
import org.springframework.stereotype.Component
import org.springframework.web.socket.TextMessage
import org.springframework.web.socket.WebSocketSession
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

@Component
class EventPhotoSessionRegistry(
    private val objectMapper: ObjectMapper,
    private val sender: WebSocketSender,
) {
    private val sessions = ConcurrentHashMap<UUID, ConcurrentHashMap<String, WebSocketSession>>()

    fun add(eventId: UUID, session: WebSocketSession) {
        val targets = sessions.computeIfAbsent(eventId) { ConcurrentHashMap() }
        targets[session.id] = sender.decorate(session) { targets.remove(it.id, it) }
    }

    fun remove(eventId: UUID, session: WebSocketSession) {
        sessions[eventId]?.remove(session.id)
    }

    fun broadcast(eventId: UUID, payload: Any) {
        val targets = sessions[eventId] ?: return
        if (targets.isEmpty()) return
        val message = TextMessage(objectMapper.writeValueAsString(payload))
        targets.values.forEach { session ->
            sender.send(session, message) { targets.remove(it.id, it) }
        }
    }

    fun sessionCount(eventId: UUID): Int = sessions[eventId]?.size ?: 0
}
