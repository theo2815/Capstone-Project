package com.quickpitik.security

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PaymentWebhookProperties
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

// HMAC-SHA256 verification for incoming payment webhooks. Provider signs the
// raw request body with the shared secret and sends the lowercase hex digest
// in the X-QuickPitik-Signature header (header name configurable). We
// re-compute over the cached raw bytes (WebhookRawBodyFilter wraps the request)
// and constant-time-compare via MessageDigest.isEqual.
//
// Anti-enumeration: every failure path (missing header, missing wrapper,
// digest mismatch) emits the same UnauthorizedException — providers learn
// nothing about which step failed.
@Component
class WebhookSignatureVerifier(
    private val properties: PaymentWebhookProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    private val secretKey: SecretKeySpec by lazy {
        SecretKeySpec(
            properties.hmacSecret.toByteArray(StandardCharsets.UTF_8),
            HMAC_ALGORITHM,
        )
    }

    fun verify(request: HttpServletRequest) {
        val signatureHeader = request.getHeader(properties.signatureHeader)?.trim()
        if (signatureHeader.isNullOrEmpty()) {
            log.warn("Webhook rejected: missing {} header", properties.signatureHeader)
            throw signatureFailure()
        }

        val cached = WebUtils.getNativeRequest(request, ContentCachingRequestWrapper::class.java)
        val rawBody = cached?.contentAsByteArray
        if (rawBody == null || rawBody.isEmpty()) {
            // Cached wrapper missing means the body was consumed before the
            // filter could capture it, or the filter scope doesn't cover this
            // path. Either way the signature can't be computed — fail closed.
            log.warn("Webhook rejected: raw body unavailable for signature check")
            throw signatureFailure()
        }

        val computed = computeSignature(rawBody).toByteArray(StandardCharsets.UTF_8)
        val provided = signatureHeader.lowercase().toByteArray(StandardCharsets.UTF_8)
        if (!MessageDigest.isEqual(computed, provided)) {
            log.warn("Webhook rejected: signature mismatch")
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

    companion object {
        private const val HMAC_ALGORITHM = "HmacSHA256"
    }
}
