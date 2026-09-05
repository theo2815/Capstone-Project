package com.quickpitik.websocket

import com.fasterxml.jackson.databind.ObjectMapper
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.any
import org.mockito.Mockito
import org.springframework.web.socket.WebSocketSession
import java.io.IOException
import java.util.UUID
import kotlin.test.assertEquals

// Spotted at the first live event (2026-09-05): under a photo firehose two
// watermark threads wrote the same WebSocket concurrently -> Tomcat rejected
// the overlapping write (TEXT_PARTIAL_WRITING), and the broad catch evicted the
// *healthy* viewer, silently dropping it from the live feed. broadcast() now
// serializes sends per session and evicts only closed / IO-failed connections.
class EventPhotoSessionRegistryTest {

    private val registry = EventPhotoSessionRegistry(ObjectMapper())
    private val eventId = UUID.randomUUID()
    private val payload = mapOf("type" to "photo.published")

    @Test
    fun `a transient send error on an open session does not evict the viewer`() {
        val session = Mockito.mock(WebSocketSession::class.java)
        Mockito.`when`(session.isOpen).thenReturn(true)
        Mockito.doThrow(IllegalStateException("TEXT_PARTIAL_WRITING"))
            .`when`(session).sendMessage(any())
        registry.add(eventId, session)

        registry.broadcast(eventId, payload)
        assertEquals(1, registry.sessionCount(eventId), "healthy session must stay subscribed after a race")

        // Once the race clears the same session keeps receiving.
        Mockito.doNothing().`when`(session).sendMessage(any())
        registry.broadcast(eventId, payload)
        assertEquals(1, registry.sessionCount(eventId))
        Mockito.verify(session, Mockito.times(2)).sendMessage(any())
    }

    @Test
    fun `a closed session is evicted`() {
        val session = Mockito.mock(WebSocketSession::class.java)
        Mockito.`when`(session.isOpen).thenReturn(false)
        registry.add(eventId, session)

        registry.broadcast(eventId, payload)

        assertEquals(0, registry.sessionCount(eventId))
        Mockito.verify(session, Mockito.never()).sendMessage(any())
    }

    @Test
    fun `an IO failure evicts the session`() {
        val session = Mockito.mock(WebSocketSession::class.java)
        Mockito.`when`(session.isOpen).thenReturn(true)
        Mockito.doThrow(IOException("broken pipe")).`when`(session).sendMessage(any())
        registry.add(eventId, session)

        registry.broadcast(eventId, payload)

        assertEquals(0, registry.sessionCount(eventId))
    }
}
