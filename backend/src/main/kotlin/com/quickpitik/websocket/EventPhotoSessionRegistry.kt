package com.quickpitik.websocket

import com.fasterxml.jackson.databind.ObjectMapper
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import org.springframework.web.socket.TextMessage
import org.springframework.web.socket.WebSocketSession
import java.io.IOException
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

@Component
class EventPhotoSessionRegistry(
    private val objectMapper: ObjectMapper,
) {
    private val log = LoggerFactory.getLogger(javaClass)
    private val sessions: ConcurrentHashMap<UUID, MutableSet<WebSocketSession>> = ConcurrentHashMap()

    fun add(eventId: UUID, session: WebSocketSession) {
        sessions.computeIfAbsent(eventId) { ConcurrentHashMap.newKeySet() }.add(session)
    }

    fun remove(eventId: UUID, session: WebSocketSession) {
        sessions[eventId]?.remove(session)
    }

    fun broadcast(eventId: UUID, payload: Any) {
        val targets = sessions[eventId] ?: return
        if (targets.isEmpty()) return
        val message = TextMessage(objectMapper.writeValueAsString(payload))
        val dead = mutableListOf<WebSocketSession>()
        targets.forEach { session ->
            if (!session.isOpen) {
                dead.add(session)
                return@forEach
            }
            try {
                // Tomcat permits only one in-flight write per session; photos finish on
                // several worker threads at once, so serialize sends to each session.
                synchronized(session) { session.sendMessage(message) }
            } catch (ex: Exception) {
                log.warn("WebSocket send failed for event {}: {}", eventId, ex.message)
                // Evict only a genuinely broken session — a transient send race on an
                // open connection must not drop a healthy viewer from the feed.
                if (!session.isOpen || ex is IOException) dead.add(session)
            }
        }
        targets.removeAll(dead.toSet())
    }

    fun sessionCount(eventId: UUID): Int = sessions[eventId]?.size ?: 0
}
