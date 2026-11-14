package com.quickpitik.service.relevance

import com.fasterxml.jackson.databind.JsonNode
import com.quickpitik.config.RelevanceProperties
import org.slf4j.LoggerFactory
import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType
import org.springframework.http.client.SimpleClientHttpRequestFactory
import org.springframework.stereotype.Component
import org.springframework.web.client.ResourceAccessException
import org.springframework.web.client.RestClient
import org.springframework.web.client.RestClientException
import org.springframework.web.client.RestClientResponseException
import java.io.IOException

// What one model call came to. A refusal or an answer cut off at max_tokens is CutOff; any error is Failed.
sealed interface LlmResult {
    data class Answer(val text: String, val model: String) : LlmResult
    data object CutOff : LlmResult
    data class Failed(val reason: String) : LlmResult
}

// RV-1.7: the Claude Messages API over plain HTTP (no SDK, P19). It never throws: the verdict code turns
// anything but an Answer into "can't tell". Called with no database transaction open.
@Component
class AnthropicClient(relevance: RelevanceProperties) {
    private val log = LoggerFactory.getLogger(javaClass)
    private val llm = relevance.llm

    val isConfigured: Boolean = llm.apiKey.isNotBlank()

    private val restClient: RestClient by lazy {
        val requestFactory = SimpleClientHttpRequestFactory().apply {
            setConnectTimeout(llm.connectTimeout)
            setReadTimeout(llm.readTimeout)
        }
        RestClient.builder()
            .requestFactory(requestFactory)
            .baseUrl(llm.baseUrl)
            .defaultHeader("x-api-key", llm.apiKey)
            .defaultHeader("anthropic-version", API_VERSION)
            .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
            .build()
    }

    fun ask(system: String, user: String, maxTokens: Int): LlmResult {
        if (!isConfigured) return LlmResult.Failed("no api key")
        val body = mapOf(
            "model" to llm.model,
            "max_tokens" to maxTokens,
            "system" to system,
            "messages" to listOf(mapOf("role" to "user", "content" to user)),
        )
        var attempt = 1
        while (true) {
            val retryable: String = try {
                val reply = restClient.post().uri("/v1/messages").body(body).retrieve().body(JsonNode::class.java)
                    ?: return LlmResult.Failed("empty response")
                return read(reply)
            } catch (ex: RestClientResponseException) {
                val status = ex.statusCode
                if (status.value() != 429 && !status.is5xxServerError) {
                    log.warn("Anthropic call refused: status={} attempt={}", status.value(), attempt)
                    return LlmResult.Failed("http ${status.value()}")
                }
                "http ${status.value()}"
            } catch (ex: RestClientException) {
                // A read timeout can surface wrapped while the reply is read, not only as ResourceAccessException.
                if (ex !is ResourceAccessException && ex.cause !is IOException) {
                    log.warn("Anthropic call failed: {} attempt={}", ex.javaClass.simpleName, attempt)
                    return LlmResult.Failed(ex.javaClass.simpleName)
                }
                "timeout or connection error"
            } catch (ex: RuntimeException) {
                log.warn("Anthropic call failed: {} attempt={}", ex.javaClass.simpleName, attempt)
                return LlmResult.Failed(ex.javaClass.simpleName)
            }
            if (attempt >= llm.maxAttempts) {
                log.warn("Anthropic call failed after {} tries: {}", attempt, retryable)
                return LlmResult.Failed("$retryable after $attempt tries")
            }
            log.info("Anthropic call retrying: {} attempt={}", retryable, attempt)
            try {
                Thread.sleep(llm.retryBackoff.toMillis())
            } catch (ex: InterruptedException) {
                Thread.currentThread().interrupt()
                return LlmResult.Failed("interrupted")
            }
            attempt++
        }
    }

    // Only end_turn is a normal finish; refusal, max_tokens and the rest are CutOff. Thinking blocks are skipped.
    private fun read(reply: JsonNode): LlmResult {
        if (reply.path("stop_reason").asText() != "end_turn") return LlmResult.CutOff
        val text = reply.path("content").filter { it.path("type").asText() == "text" }.joinToString("") { it.path("text").asText() }
        return LlmResult.Answer(text, reply.path("model").asText(llm.model))
    }

    private companion object {
        const val API_VERSION = "2026-10-21"
    }
}
