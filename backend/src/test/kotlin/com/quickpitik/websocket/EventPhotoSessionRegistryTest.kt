package com.quickpitik.websocket

import com.fasterxml.jackson.databind.ObjectMapper
import org.awaitility.Awaitility.await
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertTimeoutPreemptively
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
import org.mockito.ArgumentMatchers.any
import org.mockito.Mockito
import org.springframework.web.socket.CloseStatus
import org.springframework.web.socket.WebSocketSession
import java.io.IOException
import java.time.Duration
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// Spotted at the first live event (2026-09-05): under a photo firehose two
// watermark threads wrote the same WebSocket concurrently -> Tomcat rejected
// the overlapping write (TEXT_PARTIAL_WRITING), and the broad catch evicted the
// *healthy* viewer, silently dropping it from the live feed. broadcast() now
// isolates and serializes sends; closed, IO-failed, and over-budget sessions leave the feed.
class EventPhotoSessionRegistryTest {

    private val sender = WebSocketSender()
    private val eventId = UUID.randomUUID()
    private val payload = mapOf("type" to "photo.published")

    enum class Channel { EVENT, ME, ADMIN }

    private data class Feed(
        val add: (WebSocketSession) -> Unit,
        val remove: (WebSocketSession) -> Unit,
        val broadcast: (Any) -> Unit,
        val count: () -> Int,
    )

    private fun feed(channel: Channel): Feed = when (channel) {
        Channel.EVENT -> EventPhotoSessionRegistry(ObjectMapper(), sender).let {
            Feed({ s -> it.add(eventId, s) }, { s -> it.remove(eventId, s) },
                { p -> it.broadcast(eventId, p) }, { it.sessionCount(eventId) })
        }
        Channel.ME -> MeNotificationSessionRegistry(ObjectMapper(), sender).let {
            Feed({ s -> it.add(eventId, s) }, { s -> it.remove(eventId, s) },
                { p -> it.broadcast(eventId, p) }, { it.sessionCount(eventId) })
        }
        Channel.ADMIN -> AdminNotificationSessionRegistry(ObjectMapper(), sender).let {
            Feed(it::add, it::remove, it::broadcast, it::sessionCount)
        }
    }

    @AfterEach
    fun shutdown() {
        sender.shutdown()
    }

    private fun session(): WebSocketSession = Mockito.mock(WebSocketSession::class.java).also {
        Mockito.`when`(it.id).thenReturn(UUID.randomUUID().toString())
        Mockito.`when`(it.isOpen).thenReturn(true)
    }

    @ParameterizedTest
    @EnumSource(Channel::class)
    fun `a lone slow send cannot block broadcast or a healthy viewer and is evicted`(channel: Channel) {
        val feed = feed(channel)
        val slow = session()
        val healthy = session()
        val sending = CountDownLatch(1)
        val release = CountDownLatch(1)
        val closed = CountDownLatch(1)
        val delivered = CountDownLatch(1)
        Mockito.doAnswer {
            sending.countDown()
            release.await(5, TimeUnit.SECONDS)
            null
        }.`when`(slow).sendMessage(any())
        Mockito.doAnswer {
            closed.countDown()
            release.countDown()
            null
        }.`when`(slow).close(any())
        Mockito.doAnswer { delivered.countDown(); null }.`when`(healthy).sendMessage(any())
        feed.add(slow)
        try {
            assertTimeoutPreemptively(Duration.ofMillis(250)) { feed.broadcast(payload) }
            assertTrue(sending.await(1, TimeUnit.SECONDS))
            // The first send stays blocked while a second viewer joins.
            feed.add(healthy)
            assertTimeoutPreemptively(Duration.ofMillis(250)) { feed.broadcast(payload) }
            assertTrue(delivered.await(250, TimeUnit.MILLISECONDS), "healthy viewer must receive promptly")
            assertTrue(closed.await(1500, TimeUnit.MILLISECONDS), "one blocked write must time out without another broadcast")
            assertEquals(1, feed.count())
        } finally {
            release.countDown()
        }
    }

    @ParameterizedTest
    @EnumSource(Channel::class)
    fun `removing the original session removes its registered wrapper`(channel: Channel) {
        val feed = feed(channel)
        val session = session()
        feed.add(session)
        assertEquals(1, feed.count())

        feed.remove(session)
        feed.broadcast(payload)

        assertEquals(0, feed.count())
        Mockito.verify(session, Mockito.never()).sendMessage(any())
    }

    @ParameterizedTest
    @EnumSource(Channel::class)
    fun `broadcast during shutdown does not fail its producer`(channel: Channel) {
        val feed = feed(channel)
        val session = session()
        feed.add(session)
        sender.shutdown()

        feed.broadcast(payload)

        Mockito.verify(session, Mockito.never()).sendMessage(any())
    }

    @ParameterizedTest
    @EnumSource(Channel::class)
    fun `overflow closes and evicts a slow viewer before its write deadline`(channel: Channel) {
        val feed = feed(channel)
        val session = session()
        val sending = CountDownLatch(1)
        val release = CountDownLatch(1)
        Mockito.doAnswer {
            sending.countDown()
            release.await(5, TimeUnit.SECONDS)
            null
        }.`when`(session).sendMessage(any())
        feed.add(session)
        try {
            feed.broadcast(payload)
            assertTrue(sending.await(1, TimeUnit.SECONDS))
            feed.broadcast("x".repeat(65 * 1024))

            Mockito.verify(session, Mockito.timeout(500)).close(CloseStatus.SESSION_NOT_RELIABLE)
            assertEquals(0, feed.count())
            Mockito.verify(session, Mockito.times(1)).sendMessage(any())
        } finally {
            release.countDown()
        }
    }

    @ParameterizedTest
    @EnumSource(Channel::class)
    fun `concurrent broadcasts deliver every message without overlapping writes`(channel: Channel) {
        val feed = feed(channel)
        val session = session()
        val active = AtomicInteger()
        val peak = AtomicInteger()
        val delivered = CountDownLatch(20)
        Mockito.doAnswer {
            peak.accumulateAndGet(active.incrementAndGet(), ::maxOf)
            try {
                Thread.sleep(10)
            } finally {
                active.decrementAndGet()
                delivered.countDown()
            }
            null
        }.`when`(session).sendMessage(any())
        feed.add(session)

        repeat(20) { feed.broadcast(payload) }

        assertTrue(delivered.await(2, TimeUnit.SECONDS))
        assertEquals(1, peak.get())
        assertEquals(1, feed.count())
        Mockito.verify(session, Mockito.never()).close(any())
    }

    @Test
    fun `draining healthy buffered writes can take longer than one write deadline`() {
        val feed = feed(Channel.EVENT)
        val session = session()
        val delivered = CountDownLatch(10)
        Mockito.doAnswer {
            Thread.sleep(150)
            delivered.countDown()
            null
        }.`when`(session).sendMessage(any())
        feed.add(session)

        repeat(10) { feed.broadcast(payload) }

        assertTrue(delivered.await(3, TimeUnit.SECONDS))
        assertEquals(1, feed.count())
        Mockito.verify(session, Mockito.never()).close(any())
    }

    @ParameterizedTest
    @EnumSource(Channel::class)
    fun `a transient send error on an open session does not evict the viewer`(channel: Channel) {
        val feed = feed(channel)
        val session = session()
        Mockito.doThrow(IllegalStateException("TEXT_PARTIAL_WRITING"))
            .`when`(session).sendMessage(any())
        feed.add(session)

        feed.broadcast(payload)
        Mockito.verify(session, Mockito.timeout(1000)).sendMessage(any())
        assertEquals(1, feed.count(), "healthy session must stay subscribed after a race")

        // Once the race clears the same session keeps receiving.
        Mockito.doNothing().`when`(session).sendMessage(any())
        feed.broadcast(payload)
        Mockito.verify(session, Mockito.timeout(1000).times(2)).sendMessage(any())
        assertEquals(1, feed.count())
        Mockito.verify(session, Mockito.never()).close(any())
    }

    @ParameterizedTest
    @EnumSource(Channel::class)
    fun `a closed session is evicted`(channel: Channel) {
        val feed = feed(channel)
        val session = session()
        Mockito.`when`(session.isOpen).thenReturn(false)
        feed.add(session)

        feed.broadcast(payload)

        await().atMost(Duration.ofSeconds(2)).untilAsserted { assertEquals(0, feed.count()) }
        Mockito.verify(session, Mockito.never()).sendMessage(any())
    }

    @ParameterizedTest
    @EnumSource(Channel::class)
    fun `an IO failure evicts the session`(channel: Channel) {
        val feed = feed(channel)
        val session = session()
        Mockito.doThrow(IOException("broken pipe")).`when`(session).sendMessage(any())
        feed.add(session)

        feed.broadcast(payload)

        await().atMost(Duration.ofSeconds(2)).untilAsserted { assertEquals(0, feed.count()) }
        Mockito.verify(session, Mockito.timeout(1000)).close(CloseStatus.SESSION_NOT_RELIABLE)
    }
}
