package com.quickpitik.security

import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
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
// rejects it — fail-closed by design.
@Component
class WebhookRawBodyFilter : OncePerRequestFilter() {

    override fun shouldNotFilter(request: HttpServletRequest): Boolean =
        !request.requestURI.startsWith("/api/v1/payments/webhook/")

    override fun doFilterInternal(
        request: HttpServletRequest,
        response: HttpServletResponse,
        filterChain: FilterChain,
    ) {
        val wrapped = ContentCachingRequestWrapper(request, CONTENT_CACHE_LIMIT_BYTES)
        filterChain.doFilter(wrapped, response)
    }

    companion object {
        private const val CONTENT_CACHE_LIMIT_BYTES = 64 * 1024
    }
}
