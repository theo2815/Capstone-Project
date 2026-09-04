package com.quickpitik.service.email

import com.quickpitik.config.ResendProperties
import com.quickpitik.dto.email.ResendSendEmailRequest
import com.quickpitik.dto.email.ResendSendEmailResponse
import org.slf4j.LoggerFactory
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpStatusCode
import org.springframework.http.client.SimpleClientHttpRequestFactory
import org.springframework.stereotype.Component
import org.springframework.web.client.RestClient
import org.springframework.web.client.RestClientResponseException

// Thin REST wrapper for Resend. One endpoint suffices for v1 transactional
// receipts: POST /emails. Bearer auth (re_… key). Errors surface as
// RestClientResponseException carrying the JSON envelope Resend returns
// ({ statusCode, name, message }) — logged so misdelivers are debuggable.
@Component
class ResendClient(
    private val properties: ResendProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    private val restClient: RestClient by lazy {
        // Wires the declared-but-previously-dead timeout config — an email
        // send runs on the @Async pool, but a hung socket still pinned one of
        // its threads forever.
        val requestFactory = SimpleClientHttpRequestFactory().apply {
            setConnectTimeout(properties.connectTimeout)
            setReadTimeout(properties.readTimeout)
        }
        RestClient.builder()
            .requestFactory(requestFactory)
            .baseUrl(properties.baseUrl)
            .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer ${properties.apiKey}")
            .defaultHeader(HttpHeaders.CONTENT_TYPE, "application/json")
            .defaultHeader(HttpHeaders.ACCEPT, "application/json")
            .build()
    }

    // Retries a rate-limited or transiently-failed send instead of giving up on
    // the first 429. Resend's free tier allows ~2 requests/second, which a burst
    // of finishers checking out at the end of a race trips easily; previously
    // that 429 propagated and the receipt depended on PayMongo re-delivering the
    // webhook, which frequently never happens for an already-completed payment.
    //
    // Blocking sleeps are fine here: the only caller is OrderReceiptEmailService,
    // invoked from an AFTER_COMMIT @Async listener, so this occupies a pool
    // thread rather than the webhook's response path.
    fun send(request: ResendSendEmailRequest): ResendSendEmailResponse {
        var attempt = 0
        while (true) {
            try {
                return restClient.post()
                    .uri("/emails")
                    .body(request)
                    .retrieve()
                    .body(ResendSendEmailResponse::class.java)
                    ?: throw IllegalStateException("Resend returned empty response on POST /emails")
            } catch (ex: RestClientResponseException) {
                val status = ex.statusCode
                if (!isRetryable(status) || attempt >= MAX_ATTEMPTS - 1) {
                    log.error(
                        "Resend POST /emails failed: status={} attempts={} body={}",
                        status,
                        attempt + 1,
                        ex.responseBodyAsString,
                    )
                    throw ex
                }
                val delayMs = retryAfterMs(ex.responseHeaders?.getFirst(HttpHeaders.RETRY_AFTER))
                    ?: (BASE_BACKOFF_MS shl attempt)
                log.warn(
                    "Resend POST /emails retryable failure: status={} attempt={} retryingInMs={}",
                    status,
                    attempt + 1,
                    delayMs,
                )
                Thread.sleep(delayMs)
                attempt++
            }
        }
    }

    // Rate limits and upstream blips are worth another go; a 4xx we caused
    // (bad key, unverified sender, malformed payload) will fail identically
    // every time, so retrying it just delays the error and burns quota.
    internal fun isRetryable(status: HttpStatusCode): Boolean =
        status.value() == 429 || status.is5xxServerError

    // Resend sends Retry-After in seconds on a 429. Honour it when present, but
    // cap it — an upstream telling us to wait minutes would pin a pool thread
    // for longer than the receipt is worth, and the backoff ladder is a fine
    // fallback.
    internal fun retryAfterMs(retryAfterHeader: String?): Long? {
        val seconds = retryAfterHeader?.trim()?.toLongOrNull() ?: return null
        if (seconds <= 0) return null
        return (seconds * 1000).coerceAtMost(MAX_RETRY_AFTER_MS)
    }

    private companion object {
        const val MAX_ATTEMPTS = 3
        const val BASE_BACKOFF_MS = 500L
        const val MAX_RETRY_AFTER_MS = 5_000L
    }
}
