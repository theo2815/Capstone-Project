package com.quickpitik.security

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PaymongoProperties
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

// HMAC-SHA256 verification for PayMongo webhook deliveries. The provider
// signs `${timestamp}.${raw_body}` and sends the digest in
//
//   Paymongo-Signature: t=<unix>,te=<test_sig>,li=<live_sig>
//
// Test-mode events populate `te=`; live-mode events populate `li=`. We
// validate whichever is present — they're mutually exclusive in practice.
//
// Anti-enumeration: every failure path (missing header, missing wrapper,
// digest mismatch, bad timestamp) raises the SAME UnauthorizedException
// so a probing attacker learns nothing about which step rejected.
//
// Limitation (deferred): no timestamp drift check. A captured signature
// stays replay-able indefinitely. The application-level (provider,
// provider_ref, order_id) UNIQUE on payments and the order status guard
// in PaymongoWebhookService dedupe the most common case (same event
// re-sent). A full defense rejects requests outside a ±5-minute window;
// land alongside live-mode go-live.
@Component
class PaymongoSignatureVerifier(
    private val properties: PaymongoProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    private val secretKey: SecretKeySpec by lazy {
        SecretKeySpec(
            properties.webhookSecret.toByteArray(StandardCharsets.UTF_8),
            HMAC_ALGORITHM,
        )
    }

    fun verify(request: HttpServletRequest) {
        val signatureHeader = request.getHeader(HEADER)?.trim()
        if (signatureHeader.isNullOrEmpty()) {
            log.warn("PayMongo webhook rejected: missing {} header", HEADER)
            throw signatureFailure()
        }

        val cached = WebUtils.getNativeRequest(request, ContentCachingRequestWrapper::class.java)
        val rawBody = cached?.contentAsByteArray
        if (rawBody == null || rawBody.isEmpty()) {
            log.warn("PayMongo webhook rejected: raw body unavailable")
            throw signatureFailure()
        }

        val parts = parseSignatureHeader(signatureHeader)
        val timestamp = parts["t"]
        val provided = (parts["te"] ?: parts["li"])?.lowercase()
        if (timestamp.isNullOrEmpty() || provided.isNullOrEmpty()) {
            log.warn("PayMongo webhook rejected: signature header missing t / te / li")
            throw signatureFailure()
        }

        val payload = ByteArray(timestamp.toByteArray(StandardCharsets.UTF_8).size + 1 + rawBody.size).apply {
            val tBytes = timestamp.toByteArray(StandardCharsets.UTF_8)
            System.arraycopy(tBytes, 0, this, 0, tBytes.size)
            this[tBytes.size] = '.'.code.toByte()
            System.arraycopy(rawBody, 0, this, tBytes.size + 1, rawBody.size)
        }
        val computed = computeSignature(payload)

        if (!MessageDigest.isEqual(
                computed.toByteArray(StandardCharsets.UTF_8),
                provided.toByteArray(StandardCharsets.UTF_8),
            )
        ) {
            log.warn("PayMongo webhook rejected: signature mismatch")
            throw signatureFailure()
        }
    }

    private fun parseSignatureHeader(header: String): Map<String, String> =
        header.split(",")
            .mapNotNull { piece ->
                val eq = piece.indexOf('=')
                if (eq <= 0) null else piece.substring(0, eq).trim() to piece.substring(eq + 1).trim()
            }
            .toMap()

    private fun computeSignature(bytes: ByteArray): String {
        val mac = Mac.getInstance(HMAC_ALGORITHM).apply { init(secretKey) }
        return HexFormat.of().formatHex(mac.doFinal(bytes))
    }

    private fun signatureFailure(): UnauthorizedException = UnauthorizedException(
        code = ErrorCodes.UNAUTHORIZED,
        message = "PayMongo webhook signature header missing or invalid.",
    )

    companion object {
        const val HEADER = "Paymongo-Signature"
        private const val HMAC_ALGORITHM = "HmacSHA256"
    }
}
