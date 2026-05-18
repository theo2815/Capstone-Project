package com.quickpitik.service.email

import com.quickpitik.config.ResendProperties
import com.quickpitik.dto.email.ResendSendEmailRequest
import com.quickpitik.dto.email.ResendSendEmailResponse
import org.slf4j.LoggerFactory
import org.springframework.http.HttpHeaders
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
        RestClient.builder()
            .baseUrl(properties.baseUrl)
            .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer ${properties.apiKey}")
            .defaultHeader(HttpHeaders.CONTENT_TYPE, "application/json")
            .defaultHeader(HttpHeaders.ACCEPT, "application/json")
            .build()
    }

    fun send(request: ResendSendEmailRequest): ResendSendEmailResponse {
        return try {
            restClient.post()
                .uri("/emails")
                .body(request)
                .retrieve()
                .body(ResendSendEmailResponse::class.java)
                ?: throw IllegalStateException("Resend returned empty response on POST /emails")
        } catch (ex: RestClientResponseException) {
            log.error(
                "Resend POST /emails failed: status={} body={}",
                ex.statusCode,
                ex.responseBodyAsString,
            )
            throw ex
        }
    }
}
