package com.quickpitik.service.runner

import com.quickpitik.entity.RunnerMessage
import com.quickpitik.repository.RunnerMessageRepository
import com.quickpitik.websocket.RunnerMessageCreatedEvent
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import org.springframework.context.ApplicationEventPublisher
import java.time.Instant
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals

class RunnerRemovalNoticeTest {
    private val messages = mock(RunnerMessageRepository::class.java)
    private val publisher = mock(ApplicationEventPublisher::class.java)
    private val service = RunnerMessagesService(messages, publisher)
    private val runner = UUID.randomUUID()
    private val order = UUID.randomUUID()
    private val photo = UUID.randomUUID()
    private val request = UUID.randomUUID()
    private val now = Instant.parse("2026-10-07T00:00:00Z")
    private fun <T> arg(): T = any()

    @Test fun `a duplicate durable notice does not publish another live notification`() {
        service.pushRemovalNotice(runner, order, photo, request, now)
        verify(messages, never()).findById(arg())
        verifyNoInteractions(publisher)
    }

    @Test fun `only a newly stored notice publishes the order and privacy case`() {
        `when`(messages.insertRemovalNotice(arg(), arg(), arg(), arg(), arg(), arg(), arg())).thenReturn(1)
        val stored = RunnerMessage(runnerId = runner, kindWire = "photo_removed", body = "Privacy removal",
            orderId = order, photoRemovalRequestId = request, noticeKey = "photo_removed:$order:$photo")
        `when`(messages.findById(arg())).thenReturn(Optional.of(stored))
        var event: RunnerMessageCreatedEvent? = null
        doAnswer { event = it.getArgument(0); null }.`when`(publisher).publishEvent(arg<RunnerMessageCreatedEvent>())
        service.pushRemovalNotice(runner, order, photo, request, now)
        assertEquals(runner, event?.runnerId)
        val payload = event?.payload?.get("message") as Map<*, *>
        assertEquals("photo_removed", payload["kind"])
        assertEquals(order.toString(), payload["orderId"])
        assertEquals(request.toString(), payload["photoRemovalRequestId"])
        verify(publisher, times(1)).publishEvent(arg<RunnerMessageCreatedEvent>())
    }
}
