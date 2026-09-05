package com.quickpitik.websocket

import com.fasterxml.jackson.databind.ObjectMapper
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import org.springframework.web.socket.TextMessage
import org.springframework.web.socket.WebSocketSession
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap

// Flat session set for /ws/admin/notifications — every connected admin
// receives every broadcast (verification submitted, dispute filed, payout
// report filed). No per-key routing; the audience is "all admins" and we
// only have one bootstrap admin in v1 anyway.
@Component
class AdminNotificationSessionRegistry(
    private val objectMapper: ObjectMapper,
) {
    private val log = LoggerFactory.getLogger(javaClass)
    private val sessions: MutableSet<WebSocketSession> = ConcurrentHashMap.newKeySet()

    fun add(session: WebSocketSession) {
        sessions.add(session)
    }

    fun remove(session: WebSocketSession) {
        sessions.remove(session)
    }

    fun broadcast(payload: Any) {
        if (sessions.isEmpty()) return
        val message = TextMessage(objectMapper.writeValueAsString(payload))
        val dead = mutableListOf<WebSocketSession>()
        sessions.forEach { session ->
            if (!session.isOpen) {
                dead.add(session)
                return@forEach
            }
            try {
                // Tomcat permits only one in-flight write per session; serialize sends
                // so concurrent admin broadcasts can't collide on one connection.
                synchronized(session) { session.sendMessage(message) }
            } catch (ex: Exception) {
                log.warn("WebSocket admin send failed: {}", ex.message)
                // Evict only a genuinely broken session — a transient send race on an
                // open connection must not drop a healthy admin from the feed.
                if (!session.isOpen || ex is IOException) dead.add(session)
            }
        }
        sessions.removeAll(dead.toSet())
    }

    fun sessionCount(): Int = sessions.size
}
