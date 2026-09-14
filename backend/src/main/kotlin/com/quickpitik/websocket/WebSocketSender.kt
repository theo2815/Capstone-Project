package com.quickpitik.websocket

import jakarta.annotation.PreDestroy
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import org.springframework.web.socket.CloseStatus
import org.springframework.web.socket.WebSocketMessage
import org.springframework.web.socket.WebSocketSession
import org.springframework.web.socket.handler.ConcurrentWebSocketSessionDecorator
import org.springframework.web.socket.handler.SessionLimitExceededException
import org.springframework.web.socket.handler.WebSocketSessionDecorator
import java.io.IOException
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

@Component
class WebSocketSender {
    private val log = LoggerFactory.getLogger(javaClass)
    private val executor = Executors.newVirtualThreadPerTaskExecutor()
    // Scheduled application jobs do storage/provider I/O; deadlines must not queue behind them.
    private val deadlines = ScheduledThreadPoolExecutor(1).apply { removeOnCancelPolicy = true }

    fun decorate(session: WebSocketSession, evict: (WebSocketSession) -> Unit): WebSocketSession {
        lateinit var decorated: ConcurrentWebSocketSessionDecorator
        val timed = object : WebSocketSessionDecorator(session) {
            override fun sendMessage(message: WebSocketMessage<*>) {
                // Spring checks its time limit only on a competing send. Time each
                // actual write as well, including a lone stalled frame.
                val pending = AtomicBoolean(true)
                val timeout = deadlines.schedule({
                    if (pending.compareAndSet(true, false)) {
                        evict(decorated)
                        execute { close(decorated) }
                    }
                }, SEND_TIME_LIMIT_MS.toLong(), TimeUnit.MILLISECONDS)
                try {
                    super.sendMessage(message)
                } finally {
                    pending.set(false)
                    timeout.cancel(false)
                }
            }
        }
        decorated = ConcurrentWebSocketSessionDecorator(timed, SEND_TIME_LIMIT_MS, SEND_BUFFER_BYTES)
        return decorated
    }

    fun send(session: WebSocketSession, message: WebSocketMessage<*>, evict: (WebSocketSession) -> Unit) {
        // A blocked viewer occupies its own virtual thread, never a producer or
        // another viewer's worker. Spring serializes writes and bounds the buffer.
        // ponytail: tasks precede buffer admission; add admission limits if load tests show dispatch backlog.
        execute {
            if (!session.isOpen) {
                evict(session)
                return@execute
            }
            try {
                session.sendMessage(message)
            } catch (ex: Exception) {
                log.warn("WebSocket send failed for session {}: {}", session.id, ex.message)
                if (!session.isOpen || ex is IOException || ex is SessionLimitExceededException) {
                    evict(session)
                    close(session)
                }
            }
        }
    }

    private fun execute(action: () -> Unit) {
        try {
            executor.execute(action)
        } catch (ex: RejectedExecutionException) {
            // A shutdown racing an AFTER_COMMIT notification must not fail its producer.
            if (!executor.isShutdown) throw ex
        }
    }

    private fun close(session: WebSocketSession) {
        runCatching { session.close(CloseStatus.SESSION_NOT_RELIABLE) }
            .onFailure { log.debug("WebSocket close failed for session {}", session.id, it) }
    }

    @PreDestroy
    fun shutdown() {
        deadlines.shutdownNow()
        executor.shutdownNow()
    }

    private companion object {
        const val SEND_TIME_LIMIT_MS = 1_000
        const val SEND_BUFFER_BYTES = 64 * 1024
    }
}
