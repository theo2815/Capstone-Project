package com.quickpitik.service.orders

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.dto.orders.PaymongoCheckoutSessionRequest
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponse
import com.quickpitik.dto.orders.PaymongoPaymentIntentAttachRequest
import com.quickpitik.dto.orders.PaymongoPaymentIntentRequest
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponse
import com.quickpitik.dto.orders.PaymongoPaymentMethodRequest
import com.quickpitik.dto.orders.PaymongoPaymentMethodResponse
import com.quickpitik.dto.orders.PaymongoRefundRequest
import com.quickpitik.dto.orders.PaymongoRefundResponse
import org.slf4j.LoggerFactory
import org.springframework.http.HttpHeaders
import org.springframework.http.client.SimpleClientHttpRequestFactory
import org.springframework.stereotype.Component
import org.springframework.web.client.RestClient
import org.springframework.web.client.RestClientResponseException
import java.nio.charset.StandardCharsets
import java.util.Base64

// Thin REST wrapper for the PayMongo checkout and refund endpoints.
//
//   POST /checkout_sessions             — create
//   GET  /checkout_sessions/{id}        — retrieve/reconcile
//   POST /checkout_sessions/{id}/expire — expire abandoned checkout
//   POST/GET /refunds                    — issue/reconcile refunds
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
        // Bound provider latency so request and reconciliation threads cannot
        // hang indefinitely.
        val requestFactory = SimpleClientHttpRequestFactory().apply {
            setConnectTimeout(properties.connectTimeout)
            setReadTimeout(properties.readTimeout)
        }
        RestClient.builder()
            .requestFactory(requestFactory)
            .baseUrl(properties.baseUrl)
            .defaultHeader(HttpHeaders.AUTHORIZATION, authHeader)
            .defaultHeader(HttpHeaders.ACCEPT, "application/json")
            .defaultHeader(HttpHeaders.CONTENT_TYPE, "application/json")
            .build()
    }

    fun createCheckoutSession(
        request: PaymongoCheckoutSessionRequest,
        idempotencyKey: String,
    ): PaymongoCheckoutSessionResponse {
        return try {
            restClient.post()
                .uri("/checkout_sessions")
                .header("Idempotency-Key", idempotencyKey)
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

    fun expireCheckoutSession(id: String): PaymongoCheckoutSessionResponse {
        return try {
            restClient.post()
                .uri("/checkout_sessions/{id}/expire", id)
                .retrieve()
                .body(PaymongoCheckoutSessionResponse::class.java)
                ?: throw IllegalStateException("PayMongo returned empty response while expiring $id")
        } catch (ex: RestClientResponseException) {
            log.error(
                "PayMongo POST /checkout_sessions/{}/expire failed: status={} body={}",
                id,
                ex.statusCode,
                ex.responseBodyAsString,
            )
            throw ex
        }
    }

    fun createPaymentIntent(
        request: PaymongoPaymentIntentRequest,
        idempotencyKey: String,
    ): PaymongoPaymentIntentResponse = try {
        restClient.post()
            .uri("/payment_intents")
            .header("Idempotency-Key", idempotencyKey)
            .body(request)
            .retrieve()
            .body(PaymongoPaymentIntentResponse::class.java)
            ?: throw IllegalStateException("PayMongo returned empty response on POST /payment_intents")
    } catch (ex: RestClientResponseException) {
        log.error("PayMongo POST /payment_intents failed: status={} body={}", ex.statusCode, ex.responseBodyAsString)
        throw ex
    }

    fun retrievePaymentIntent(id: String): PaymongoPaymentIntentResponse = try {
        restClient.get()
            .uri("/payment_intents/{id}", id)
            .retrieve()
            .body(PaymongoPaymentIntentResponse::class.java)
            ?: throw IllegalStateException("PayMongo returned empty response on GET /payment_intents/$id")
    } catch (ex: RestClientResponseException) {
        log.error("PayMongo GET /payment_intents/{} failed: status={} body={}", id, ex.statusCode, ex.responseBodyAsString)
        throw ex
    }

    fun createPaymentMethod(request: PaymongoPaymentMethodRequest): PaymongoPaymentMethodResponse = try {
        restClient.post()
            .uri("/payment_methods")
            .body(request)
            .retrieve()
            .body(PaymongoPaymentMethodResponse::class.java)
            ?: throw IllegalStateException("PayMongo returned empty response on POST /payment_methods")
    } catch (ex: RestClientResponseException) {
        log.error("PayMongo POST /payment_methods failed: status={} body={}", ex.statusCode, ex.responseBodyAsString)
        throw ex
    }

    fun attachPaymentMethod(
        paymentIntentId: String,
        request: PaymongoPaymentIntentAttachRequest,
    ): PaymongoPaymentIntentResponse = try {
        restClient.post()
            .uri("/payment_intents/{id}/attach", paymentIntentId)
            .body(request)
            .retrieve()
            .body(PaymongoPaymentIntentResponse::class.java)
            ?: throw IllegalStateException("PayMongo returned empty response while attaching $paymentIntentId")
    } catch (ex: RestClientResponseException) {
        log.error(
            "PayMongo POST /payment_intents/{}/attach failed: status={} body={}",
            paymentIntentId,
            ex.statusCode,
            ex.responseBodyAsString,
        )
        throw ex
    }

    fun createRefund(request: PaymongoRefundRequest, idempotencyKey: String): PaymongoRefundResponse {
        return try {
            restClient.post()
                .uri("/refunds")
                .header("Idempotency-Key", idempotencyKey)
                .body(request)
                .retrieve()
                .body(PaymongoRefundResponse::class.java)
                ?: throw IllegalStateException("PayMongo returned empty response on POST /refunds")
        } catch (ex: RestClientResponseException) {
            log.error("PayMongo POST /refunds failed: status={} body={}", ex.statusCode, ex.responseBodyAsString)
            throw ex
        }
    }

    fun retrieveRefund(id: String): PaymongoRefundResponse {
        return try {
            restClient.get()
                .uri("/refunds/{id}", id)
                .retrieve()
                .body(PaymongoRefundResponse::class.java)
                ?: throw IllegalStateException("PayMongo returned empty response on GET /refunds/$id")
        } catch (ex: RestClientResponseException) {
            log.error("PayMongo GET /refunds/{} failed: status={} body={}", id, ex.statusCode, ex.responseBodyAsString)
            throw ex
        }
    }
}
