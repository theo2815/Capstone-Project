package com.quickpitik.service.ai

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RestClientConfig
import com.sun.net.httpserver.HttpServer
import java.net.InetSocketAddress
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertAll
import java.time.Duration
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertIs

class AiApiFaceWriteRetryTest {
    @Test fun `timed out enrollment is never automatically resubmitted`() = checkNoReplay(false)
    @Test fun `timed out face mega submission is never automatically resubmitted`() = checkNoReplay(true)
    @Test fun `unavailable enrollment is not replayed by the HTTP transport`() = checkNoReplay(false, false)
    @Test fun `unavailable face mega is not replayed by the HTTP transport`() = checkNoReplay(true, false)

    private fun checkNoReplay(mega: Boolean, timeout: Boolean = true) {
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        val executor = Executors.newCachedThreadPool()
        val requestCount = AtomicInteger()
        try {
            val body = if (mega) """{"success":true,"data":{"job_id":"job-one"}}"""
                else """{"success":true,"data":{"person_id":"person-one"}}"""
            server.executor = executor
            server.createContext("/") { exchange ->
                exchange.use {
                    it.requestBody.readAllBytes()
                    val first = requestCount.incrementAndGet() == 1
                    if (first && timeout) Thread.sleep(400)
                    runCatching {
                        it.responseHeaders.add("Content-Type", "application/json")
                        it.sendResponseHeaders(if (first && !timeout) 503 else 200, body.toByteArray().size.toLong())
                        it.responseBody.write(body.toByteArray())
                    }
                }
            }
            server.start()
            val props = AiApiProperties(baseUrl = "http://127.0.0.1:${server.address.port}", readTimeout = Duration.ofMillis(80),
                maxRetries = 2, backoffBaseMillis = 0)
            val client = AiApiClient(RestClientConfig().aiApiRestClient(props), props, jacksonObjectMapper())
            val outcome = runCatching {
                if (mega) client.facesEnrollMega(listOf(NamedFile("photo", byteArrayOf(1), "image/jpeg")), UUID.randomUUID())
                else client.facesEnroll(byteArrayOf(1), "image/jpeg", "photo.jpg", "photo", null, UUID.randomUUID())
            }
            assertAll(
                { assertIs<AiApiException>(outcome.exceptionOrNull()) },
                { assertEquals(1, requestCount.get()) },
            )
        } finally {
            server.stop(0)
            executor.shutdownNow()
        }
    }
}
