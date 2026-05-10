package com.quickpitik.security

import com.quickpitik.config.PaymentWebhookProperties
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

// Pure-unit regression net for C-2 (HMAC verification) and the docblock /
// log additions of D-3 / D-4. Mirrors the existing IdempotencyKeyTest /
// PaginationParamsTest convention: JUnit5 + kotlin.test, no Spring context,
// no @SpringBootTest. The verifier only depends on PaymentWebhookProperties
// (a data class) and an HttpServletRequest that exposes a
// ContentCachingRequestWrapper, which we hand-build via Spring's
// MockHttpServletRequest (already on the classpath via spring-security-test).
class WebhookSignatureVerifierTest {

    private val secret = "test-secret-DO-NOT-USE-IN-PROD"
    private val properties = PaymentWebhookProperties(
        hmacSecret = secret,
        signatureHeader = "X-QuickPitik-Signature",
    )
    private val verifier = WebhookSignatureVerifier(properties)

    @Test
    fun `verify accepts request with matching signature`() {
        val body = """{"orderId":"abc","status":"SUCCEEDED"}""".toByteArray(StandardCharsets.UTF_8)
        val sig = signBody(secret, body)
        verifier.verify(wrappedRequest(body, sig))
    }

    @Test
    fun `verify rejects when signature header is missing`() {
        val body = """{"orderId":"abc"}""".toByteArray(StandardCharsets.UTF_8)
        assertThrows<UnauthorizedException> {
            verifier.verify(wrappedRequest(body, signature = null))
        }
    }

    @Test
    fun `verify rejects when signature header is whitespace only`() {
        val body = """{"orderId":"abc"}""".toByteArray(StandardCharsets.UTF_8)
        assertThrows<UnauthorizedException> {
            verifier.verify(wrappedRequest(body, signature = "   "))
        }
    }

    @Test
    fun `verify rejects when signature is wrong digest`() {
        val body = """{"orderId":"abc"}""".toByteArray(StandardCharsets.UTF_8)
        val wrongSig = "deadbeef".repeat(8)  // 64 hex chars = same length as a real SHA-256 digest
        assertThrows<UnauthorizedException> {
            verifier.verify(wrappedRequest(body, signature = wrongSig))
        }
    }

    @Test
    fun `verify rejects when body is tampered after signing`() {
        val signedBody = """{"orderId":"abc"}""".toByteArray(StandardCharsets.UTF_8)
        val tamperedBody = """{"orderId":"xyz"}""".toByteArray(StandardCharsets.UTF_8)
        val sig = signBody(secret, signedBody)
        assertThrows<UnauthorizedException> {
            verifier.verify(wrappedRequest(tamperedBody, sig))
        }
    }

    @Test
    fun `verify accepts signature with surrounding whitespace`() {
        val body = """{"x":1}""".toByteArray(StandardCharsets.UTF_8)
        val sig = signBody(secret, body)
        verifier.verify(wrappedRequest(body, signature = "  $sig  "))
    }

    @Test
    fun `verify accepts signature in uppercase hex`() {
        val body = """{"x":1}""".toByteArray(StandardCharsets.UTF_8)
        val sig = signBody(secret, body).uppercase()
        verifier.verify(wrappedRequest(body, sig))
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private fun signBody(secret: String, body: ByteArray): String {
        val mac = Mac.getInstance("HmacSHA256").apply {
            init(SecretKeySpec(secret.toByteArray(StandardCharsets.UTF_8), "HmacSHA256"))
        }
        return HexFormat.of().formatHex(mac.doFinal(body))
    }

    private fun wrappedRequest(rawBody: ByteArray, signature: String?): HttpServletRequest {
        val mock = MockHttpServletRequest("POST", "/api/v1/payments/webhook/test").apply {
            setContent(rawBody)
            signature?.let { addHeader("X-QuickPitik-Signature", it) }
        }
        // Two-arg constructor mirrors production (WebhookRawBodyFilter:33) and
        // sidesteps the deprecation on the single-arg form. 64 KB cap is the
        // production default; test bodies are tens of bytes.
        val wrapped = ContentCachingRequestWrapper(mock, 64 * 1024)
        // Drain the stream so ContentCachingRequestWrapper populates cachedContent —
        // production reaches the same state via Jackson's @RequestBody read.
        wrapped.inputStream.readAllBytes()
        return wrapped
    }
}
