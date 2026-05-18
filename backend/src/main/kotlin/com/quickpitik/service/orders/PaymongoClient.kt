package com.quickpitik.service.orders

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.dto.orders.PaymongoCheckoutSessionRequest
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponse
import org.slf4j.LoggerFactory
import org.springframework.http.HttpHeaders
import org.springframework.stereotype.Component
import org.springframework.web.client.RestClient
import org.springframework.web.client.RestClientResponseException
import java.nio.charset.StandardCharsets
import java.util.Base64

// Thin REST wrapper for PayMongo. Two endpoints suffice for v1:
//
//   POST /checkout_sessions             — create
//   GET  /checkout_sessions/{id}        — retrieve (used on idempotent replay
//                                          to recover the checkout_url)
//
// PayMongo uses HTTP Basic with the secret key as the username and empty
// password — no OAuth, no bearer. Errors come back as JSON envelopes
// `{ errors: [{ code, detail, source }] }`; the wrapping
// RestClientResponseException carries the body so OrderService can log it.
@Component
class PaymongoClient(
    private val properties: PaymongoProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    private val authHeader: String by lazy {
        val raw = "${properties.secretKey}:"
        "Basic " + Base64.getEncoder().encodeToString(raw.toByteArray(StandardCharsets.UTF_8))
    }

    private val restClient: RestClient by lazy {
        RestClient.builder()
            .baseUrl(properties.baseUrl)
            .defaultHeader(HttpHeaders.AUTHORIZATION, authHeader)
            .defaultHeader(HttpHeaders.ACCEPT, "application/json")
            .defaultHeader(HttpHeaders.CONTENT_TYPE, "application/json")
            .build()
    }

    fun createCheckoutSession(
        request: PaymongoCheckoutSessionRequest,
    ): PaymongoCheckoutSessionResponse {
        return try {
            restClient.post()
                .uri("/checkout_sessions")
                .body(request)
                .retrieve()
                .body(PaymongoCheckoutSessionResponse::class.java)
                ?: throw IllegalStateException("PayMongo returned empty response on POST /checkout_sessions")
        } catch (ex: RestClientResponseException) {
            log.error(
                "PayMongo POST /checkout_sessions failed: status={} body={}",
                ex.statusCode,
                ex.responseBodyAsString,
            )
            throw ex
        }
    }

    fun retrieveCheckoutSession(id: String): PaymongoCheckoutSessionResponse {
        return try {
            restClient.get()
                .uri("/checkout_sessions/{id}", id)
                .retrieve()
                .body(PaymongoCheckoutSessionResponse::class.java)
                ?: throw IllegalStateException("PayMongo returned empty response on GET /checkout_sessions/$id")
        } catch (ex: RestClientResponseException) {
            log.error(
                "PayMongo GET /checkout_sessions/{} failed: status={} body={}",
                id,
                ex.statusCode,
                ex.responseBodyAsString,
            )
            throw ex
        }
    }
}
