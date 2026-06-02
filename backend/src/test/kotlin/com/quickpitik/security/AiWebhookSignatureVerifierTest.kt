package com.quickpitik.security

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.AiWebhookProperties
import com.quickpitik.exception.UnauthorizedException
import jakarta.servlet.http.HttpServletRequest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.springframework.mock.web.MockHttpServletRequest
import org.springframework.web.util.ContentCachingRequestWrapper
import java.nio.charset.StandardCharsets
import java.util.HexFormat
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

// Pure-unit net for the Phase-C ai-api webhook verifier. Mirrors
// WebhookSignatureVerifierTest. The key difference from the payment verifier is
// that ai-api sends the digest as "sha256=<hex>" — the verifier must strip that
// prefix before comparing. Also asserts the fail-closed behaviour when no secret
// is configured (the receiver must never accept unauthenticated callbacks).
class AiWebhookSignatureVerifierTest {

    private val secret = "ai-webhook-secret-DO-NOT-USE-IN-PROD"

    private fun verifier(withSecret: String = secret): AiWebhookSignatureVerifier =
        AiWebhookSignatureVerifier(
            AiApiProperties(webhook = AiWebhookProperties(enabled = true, hmacSecret = withSecret)),
        )

    @Test
    fun `verify accepts a sha256-prefixed signature (ai-api's wire format)`() {
        val body = """{"event":"job.completed","job_id":"j1","result_count":3}""".toByteArray(StandardCharsets.UTF_8)
        val sig = "sha256=" + signBody(secret, body)
        verifier().verify(wrappedRequest(body, sig))
    }

    @Test
    fun `verify accepts a bare hex signature (no prefix)`() {
        val body = """{"event":"job.completed","job_id":"j1"}""".toByteArray(StandardCharsets.UTF_8)
        verifier().verify(wrappedRequest(body, signBody(secret, body)))
    }

    @Test
    fun `verify accepts an uppercase prefixed signature`() {
        val body = """{"event":"job.failed","job_id":"j2"}""".toByteArray(StandardCharsets.UTF_8)
        val sig = "sha256=" + signBody(secret, body).uppercase()
        verifier().verify(wrappedRequest(body, sig))
    }

    @Test
    fun `verify rejects a missing header`() {
        val body = """{"job_id":"j1"}""".toByteArray(StandardCharsets.UTF_8)
        assertThrows<UnauthorizedException> { verifier().verify(wrappedRequest(body, signature = null)) }
    }

    @Test
    fun `verify rejects a wrong digest`() {
        val body = """{"job_id":"j1"}""".toByteArray(StandardCharsets.UTF_8)
        val wrong = "sha256=" + "deadbeef".repeat(8)
        assertThrows<UnauthorizedException> { verifier().verify(wrappedRequest(body, wrong)) }
    }

    @Test
    fun `verify rejects a body tampered after signing`() {
        val signed = """{"job_id":"j1"}""".toByteArray(StandardCharsets.UTF_8)
        val tampered = """{"job_id":"j2"}""".toByteArray(StandardCharsets.UTF_8)
        val sig = "sha256=" + signBody(secret, signed)
        assertThrows<UnauthorizedException> { verifier().verify(wrappedRequest(tampered, sig)) }
    }

    @Test
    fun `verify fails closed when no secret is configured`() {
        val body = """{"job_id":"j1"}""".toByteArray(StandardCharsets.UTF_8)
        // A blank-secret verifier must reject before it even computes a digest —
        // the signature value is irrelevant, so a normally-signed one still fails.
        val sig = "sha256=" + signBody(secret, body)
        assertThrows<UnauthorizedException> { verifier(withSecret = "").verify(wrappedRequest(body, sig)) }
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private fun signBody(secret: String, body: ByteArray): String {
        val mac = Mac.getInstance("HmacSHA256").apply {
            init(SecretKeySpec(secret.toByteArray(StandardCharsets.UTF_8), "HmacSHA256"))
        }
        return HexFormat.of().formatHex(mac.doFinal(body))
    }

    private fun wrappedRequest(rawBody: ByteArray, signature: String?): HttpServletRequest {
        val mock = MockHttpServletRequest("POST", "/api/v1/internal/ai-webhooks").apply {
            setContent(rawBody)
            signature?.let { addHeader("X-QuickPitik-Signature", it) }
        }
        val wrapped = ContentCachingRequestWrapper(mock, 64 * 1024)
        wrapped.inputStream.readAllBytes()
        return wrapped
    }
}
