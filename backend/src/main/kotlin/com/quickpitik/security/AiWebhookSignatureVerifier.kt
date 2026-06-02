package com.quickpitik.security

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.exception.UnauthorizedException
import jakarta.servlet.http.HttpServletRequest
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import org.springframework.web.util.ContentCachingRequestWrapper
import org.springframework.web.util.WebUtils
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.HexFormat
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

// HMAC-SHA256 verification for inbound ai-api job webhooks. ai-api signs the raw
// JSON body with the secret the backend registered and sends the digest as
// "sha256=<hex>" in X-QuickPitik-Signature. We strip the prefix, recompute over
// the cached raw bytes (WebhookRawBodyFilter wraps the request) and constant-
// time-compare. Distinct from the payment verifier: different secret, and ai-api
// uses the "sha256=" prefix the payment provider does not. Fail-closed on every
// path; an unsigned or stale-secret delivery raises 401 before the service layer.
@Component
class AiWebhookSignatureVerifier(
    private val aiApiProperties: AiApiProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    private val secretKey: SecretKeySpec by lazy {
        SecretKeySpec(
            aiApiProperties.webhook.hmacSecret.toByteArray(StandardCharsets.UTF_8),
            HMAC_ALGORITHM,
        )
    }

    fun verify(request: HttpServletRequest) {
        if (aiApiProperties.webhook.hmacSecret.isBlank()) {
            // No secret configured means the receiver isn't meant to be live —
            // refuse rather than accept unauthenticated callbacks.
            log.warn("ai webhook rejected: no HMAC secret configured")
            throw signatureFailure()
        }

        val header = request.getHeader(SIGNATURE_HEADER)?.trim()
        if (header.isNullOrEmpty()) {
            log.warn("ai webhook rejected: missing {} header", SIGNATURE_HEADER)
            throw signatureFailure()
        }

        val cached = WebUtils.getNativeRequest(request, ContentCachingRequestWrapper::class.java)
        val rawBody = cached?.contentAsByteArray
        if (rawBody == null || rawBody.isEmpty()) {
            log.warn("ai webhook rejected: raw body unavailable for signature check")
            throw signatureFailure()
        }

        val provided = header.removePrefix("sha256=").lowercase().toByteArray(StandardCharsets.UTF_8)
        val computed = computeSignature(rawBody).toByteArray(StandardCharsets.UTF_8)
        if (!MessageDigest.isEqual(computed, provided)) {
            log.warn("ai webhook rejected: signature mismatch")
            throw signatureFailure()
        }
    }

    private fun computeSignature(body: ByteArray): String {
        val mac = Mac.getInstance(HMAC_ALGORITHM).apply { init(secretKey) }
        return HexFormat.of().formatHex(mac.doFinal(body))
    }

    private fun signatureFailure(): UnauthorizedException = UnauthorizedException(
        code = ErrorCodes.UNAUTHORIZED,
        message = "Webhook signature header missing or invalid.",
    )

    private companion object {
        const val HMAC_ALGORITHM = "HmacSHA256"
        const val SIGNATURE_HEADER = "X-QuickPitik-Signature"
    }
}
