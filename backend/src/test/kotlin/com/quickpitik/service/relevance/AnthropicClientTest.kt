package com.quickpitik.service.relevance

import ch.qos.logback.classic.Logger
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.read.ListAppender
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.config.RelevanceProperties
import com.sun.net.httpserver.HttpServer
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.slf4j.LoggerFactory
import java.net.InetSocketAddress
import java.time.Duration
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue

class AnthropicClientTest {

    private data class Reply(val status: Int, val body: String, val delayMs: Long = 0)
    private data class Seen(val headers: Map<String, String?>, val body: String)

    private val executor = Executors.newCachedThreadPool()
    private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
    private val seen = CopyOnWriteArrayList<Seen>()
    private var replies = listOf<Reply>()

    private fun serve(vararg r: Reply) {
        replies = r.toList()
        server.executor = executor
        server.createContext("/v1/messages") { ex ->
            ex.use {
                val body = String(it.requestBody.readAllBytes())
                seen += Seen(listOf("x-api-key", "anthropic-version", "content-type").associateWith { h -> it.requestHeaders.getFirst(h) }, body)
                val reply = replies[minOf(seen.size, replies.size) - 1]
                if (reply.delayMs > 0) Thread.sleep(reply.delayMs)
                runCatching {
                    val bytes = reply.body.toByteArray()
                    it.responseHeaders.add("Content-Type", "application/json")
                    it.sendResponseHeaders(reply.status, bytes.size.toLong())
                    it.responseBody.write(bytes)
                }
            }
        }
        server.start()
    }

    @AfterEach fun stop() {
        server.stop(0)
        executor.shutdownNow()
    }

    private fun client(key: String = KEY, readTimeout: Duration = Duration.ofSeconds(5)) = AnthropicClient(
        RelevanceProperties(llm = RelevanceProperties.Llm(apiKey = key, baseUrl = "http://127.0.0.1:${server.address.port}",
            model = "claude-sonnet-5-5", readTimeout = readTimeout, maxAttempts = 3, retryBackoff = Duration.ZERO)),
    )

    private fun answer(text: String, stopReason: String = "end_turn") = Reply(200, jacksonObjectMapper().writeValueAsString(mapOf(
        "id" to "msg_1", "type" to "message", "role" to "assistant", "model" to "claude-sonnet-5-5",
        "content" to listOf(mapOf("type" to "thinking", "thinking" to "", "signature" to "s"), mapOf("type" to "text", "text" to text)),
        "stop_reason" to stopReason)))

    private val overloaded = Reply(429, """{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}""")

    @Test fun `it sends the key, the model and both texts`() {
        serve(answer("""{"verdict":"fits"}"""))

        val result = client().ask("the system text", "the user text", 512)

        assertEquals(LlmResult.Answer("""{"verdict":"fits"}""", "claude-sonnet-5-5"), result)
        val req = seen.single()
        assertEquals(KEY, req.headers["x-api-key"])
        assertEquals("2026-10-21", req.headers["anthropic-version"])
        val body = jacksonObjectMapper().readTree(req.body)
        assertEquals("claude-sonnet-5-5", body["model"].asText())
        assertEquals(512, body["max_tokens"].asInt())
        assertEquals("the system text", body["system"].asText())
        assertEquals("user", body["messages"][0]["role"].asText())
        assertEquals("the user text", body["messages"][0]["content"].asText())
        assertEquals(1, body["messages"].size())
    }

    @Test fun `a 429 is retried and the second answer is returned`() {
        serve(overloaded, answer("second"))

        assertEquals(LlmResult.Answer("second", "claude-sonnet-5-5"), client().ask("s", "u", 100))
        assertEquals(2, seen.size)
    }

    @Test fun `after the last try it reports failed`() {
        serve(Reply(529, """{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"""))

        assertIs<LlmResult.Failed>(client().ask("s", "u", 100))
        assertEquals(3, seen.size)
    }

    @Test fun `a slow answer times out and is retried`() {
        serve(answer("late").copy(delayMs = 600), answer("on time"))

        assertEquals(LlmResult.Answer("on time", "claude-sonnet-5-5"), client(readTimeout = Duration.ofMillis(150)).ask("s", "u", 100))
        assertEquals(2, seen.size)
    }

    @Test fun `a 400 is not retried`() {
        serve(Reply(400, """{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"""))

        assertIs<LlmResult.Failed>(client().ask("s", "u", 100))
        assertEquals(1, seen.size)
    }

    @Test fun `an answer that did not end normally is cut off`() {
        serve(answer("""{"verdict":""", stopReason = "max_tokens"))
        assertEquals(LlmResult.CutOff, client().ask("s", "u", 100))

        replies = listOf(answer("", stopReason = "refusal"))
        assertEquals(LlmResult.CutOff, client().ask("s", "u", 100))
        assertEquals(2, seen.size, "a cut-off answer is not retried")
    }

    @Test fun `with no key it is not configured and sends nothing`() {
        serve(answer("never"))
        val c = client(key = " ")

        assertFalse(c.isConfigured)
        assertIs<LlmResult.Failed>(c.ask("s", "u", 100))
        assertEquals(0, seen.size)
        assertTrue(client().isConfigured)
    }

    @Test fun `the key never appears in a log line`() {
        val logger = LoggerFactory.getLogger(AnthropicClient::class.java) as Logger
        val appender = ListAppender<ILoggingEvent>().apply { start() }
        logger.addAppender(appender)
        try {
            serve(overloaded, Reply(401, """{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"""))
            val c = client()
            c.ask("s", "u", 100)

            assertTrue(appender.list.isNotEmpty(), "the failures were logged")
            appender.list.forEach { assertFalse(KEY in it.formattedMessage, "key in: ${it.formattedMessage}") }
            assertFalse(KEY in RelevanceProperties(llm = RelevanceProperties.Llm(apiKey = KEY)).toString())
        } finally {
            logger.detachAppender(appender)
        }
    }

    private companion object {
        const val KEY = "sk-ant-test-SECRET-1234"
    }
}
