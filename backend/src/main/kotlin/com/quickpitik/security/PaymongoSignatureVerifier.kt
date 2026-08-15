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
import java.time.Instant
import java.util.HexFormat
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlin.math.abs

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
// Anti-replay: deliveries outside a ±5-minute window are rejected, so a
// captured signature stops being useful once the window closes (previously it
// stayed replay-able indefinitely — D-3). This layers on top of the
// application-level (provider, provider_ref, order_id) UNIQUE on payments and
// the order status guard in PaymongoWebhookService, which already deduped the
// common case of the same event being re-sent.
//
// The window is checked only AFTER the HMAC matches: an attacker without the
// secret can't use it to probe for a valid timestamp format, and a genuine
// delivery delayed past the window is the only thing that ever reaches it.
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

        // The signature is authentic; now make sure it isn't a replay of an old
        // one. A non-numeric `t` can't have produced a valid HMAC in practice,
        // but treat it as a rejection rather than trusting it.
        val sentAtEpochSeconds = timestamp.toLongOrNull()
        if (sentAtEpochSeconds == null) {
            log.warn("PayMongo webhook rejected: non-numeric timestamp")
            throw signatureFailure()
        }
        val skewSeconds = abs(Instant.now().epochSecond - sentAtEpochSeconds)
        if (skewSeconds > MAX_CLOCK_SKEW_SECONDS) {
            log.warn(
                "PayMongo webhook rejected: timestamp outside the replay window (skew={}s, max={}s)",
                skewSeconds,
                MAX_CLOCK_SKEW_SECONDS,
            )
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

        // ±5 minutes, the same window Stripe and PayMongo document. Wide enough
        // to absorb ordinary clock drift between their senders and our host,
        // narrow enough that a captured delivery expires quickly.
        private const val MAX_CLOCK_SKEW_SECONDS = 300L
    }
}
