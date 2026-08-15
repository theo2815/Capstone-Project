package com.quickpitik.security

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.exception.UnauthorizedException
import jakarta.servlet.http.HttpServletRequest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.springframework.mock.web.MockHttpServletRequest
import org.springframework.web.util.ContentCachingRequestWrapper
import java.nio.charset.StandardCharsets
import java.time.Instant
import java.util.HexFormat
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

// Closes D-3, deferred since 2026-05-10: the verifier checked the HMAC but
// never the `t=` value, so a captured delivery stayed replay-able forever.
//
// The signature must still be authentic for the window to be reached at all —
// these tests sign every payload properly and vary only the timestamp, which
// is the distinction the window is there to make.
class PaymongoSignatureVerifierReplayTest {

    private val secret = "whsk_test-secret-DO-NOT-USE-IN-PROD"
    private val verifier = PaymongoSignatureVerifier(PaymongoProperties(webhookSecret = secret))
    private val body = """{"data":{"attributes":{"type":"checkout_session.payment.paid"}}}"""
        .toByteArray(StandardCharsets.UTF_8)

    @Test
    fun `a delivery signed just now is accepted`() {
        verifier.verify(signedRequest(atEpochSeconds = now()))
    }

    @Test
    fun `a replay from an hour ago is rejected despite a valid signature`() {
        assertThrows<UnauthorizedException> {
            verifier.verify(signedRequest(atEpochSeconds = now() - 3600))
        }
    }

    @Test
    fun `a delivery inside the five minute window is accepted`() {
        // Ordinary clock drift and PayMongo's own retry delay both land here.
        verifier.verify(signedRequest(atEpochSeconds = now() - 240))
    }

    @Test
    fun `a delivery just outside the five minute window is rejected`() {
        assertThrows<UnauthorizedException> {
            verifier.verify(signedRequest(atEpochSeconds = now() - 301))
        }
    }

    @Test
    fun `a timestamp from the future beyond the window is rejected`() {
        // Skew is absolute — a sender whose clock runs far ahead is as
        // suspicious as one running far behind.
        assertThrows<UnauthorizedException> {
            verifier.verify(signedRequest(atEpochSeconds = now() + 3600))
        }
    }

    @Test
    fun `a non-numeric timestamp is rejected rather than trusted`() {
        assertThrows<UnauthorizedException> {
            verifier.verify(signedRequest(atEpochSeconds = null, rawTimestamp = "not-a-number"))
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private fun now(): Long = Instant.now().epochSecond

    // Builds a request whose `te=` digest genuinely matches `${t}.${body}`, so
    // the only thing under test is the replay window.
    private fun signedRequest(atEpochSeconds: Long?, rawTimestamp: String? = null): HttpServletRequest {
        val t = rawTimestamp ?: atEpochSeconds.toString()
        val payload = "$t.".toByteArray(StandardCharsets.UTF_8) + body
        val mac = Mac.getInstance("HmacSHA256").apply {
            init(SecretKeySpec(secret.toByteArray(StandardCharsets.UTF_8), "HmacSHA256"))
        }
        val signature = HexFormat.of().formatHex(mac.doFinal(payload))

        val mock = MockHttpServletRequest("POST", "/api/v1/payments/webhook/paymongo").apply {
            setContent(body)
            addHeader(PaymongoSignatureVerifier.HEADER, "t=$t,te=$signature")
        }
        val wrapped = ContentCachingRequestWrapper(mock, 64 * 1024)
        wrapped.inputStream.readAllBytes()
        return wrapped
    }
}
