package com.quickpitik.security

import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import org.springframework.web.filter.OncePerRequestFilter
import org.springframework.web.util.ContentCachingRequestWrapper

// Wraps payment-webhook requests so the raw JSON body remains accessible AFTER
// Jackson deserialization. HMAC must be computed over the provider-signed wire
// bytes — re-serializing the parsed DTO would diverge on whitespace and key
// ordering. Scoped to /api/v1/payments/webhook/** so the rest of the API does
// not pay the per-request wrapping cost.
//
// Cache cap (CONTENT_CACHE_LIMIT_BYTES) bounds in-memory buffering against
// hostile oversized bodies. Real provider payloads are <2 KB; 64 KB is more
// than enough headroom while denying the trivial memory-exhaustion vector.
// A body larger than the cap will have its cached bytes truncated, so the
// computed HMAC will not match the provider's signature and the verifier
// rejects it — fail-closed by design. We log a WARN when the cap is hit so
// the silent rejection is debuggable: a legitimate provider sending oversized
// payloads is the operator's signal to bump the cap or review provider
// config; without the log the failure mode looks identical to "wrong secret"
// from the outside.
@Component
class WebhookRawBodyFilter : OncePerRequestFilter() {

    private val log = LoggerFactory.getLogger(javaClass)

    override fun shouldNotFilter(request: HttpServletRequest): Boolean =
        !request.requestURI.startsWith("/api/v1/payments/webhook/")

    override fun doFilterInternal(
        request: HttpServletRequest,
        response: HttpServletResponse,
        filterChain: FilterChain,
    ) {
        val wrapped = ContentCachingRequestWrapper(request, CONTENT_CACHE_LIMIT_BYTES)
        filterChain.doFilter(wrapped, response)
        if (wrapped.contentAsByteArray.size >= CONTENT_CACHE_LIMIT_BYTES) {
            log.warn(
                "Webhook raw body hit the {} byte cap on {} — HMAC will not match. " +
                    "Consider bumping CONTENT_CACHE_LIMIT_BYTES or reviewing provider payload size.",
                CONTENT_CACHE_LIMIT_BYTES,
                request.requestURI,
            )
        }
    }

    companion object {
        private const val CONTENT_CACHE_LIMIT_BYTES = 64 * 1024
    }
}
