package com.quickpitik.security

import com.quickpitik.support.PostgresIntegrationTest
import org.junit.jupiter.api.Test
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.test.context.ActiveProfiles
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import java.util.HexFormat
import kotlin.test.assertEquals

@SpringBootTest(
    webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties = [
        "jwt.secret=test-jwt-secret-long-enough-for-hs256-signing",
        "app.platform.order-capability-secret=test-capability-secret-0123456789abcdef",
        "app.watermark.seed-secret=test-watermark-secret-0123456789abcdef",
        "app.webhooks.payment.hmac-secret=test-payment-webhook-secret",
        "app.payments.paymongo.secret-key=sk_test_valid",
        "app.payments.paymongo.webhook-secret=whsk_test_valid",
        "app.email.resend.api-key=re_test_valid",
        "app.admin.bootstrap-email=",
        "app.admin.bootstrap-password=test-strong-admin-password",
        "app.storage.backend=S3",
        "app.storage.endpoint=https://storage.example.invalid",
        "app.storage.access-key=test-access-key",
        "app.storage.secret-key=test-secret-key",
        "app.cors.allowed-origins=https://quickpitik.com",
        "app.security.origin-secret=test-origin-secret-0123456789abcdef",
    ],
)
@ActiveProfiles(profiles = ["prod"], inheritProfiles = false)
class OriginSecretIntegrationTest : PostgresIntegrationTest() {
    @LocalServerPort private var port = 0
    private val client = HttpClient.newHttpClient()

    @Test
    fun `direct requests fail before routes or auth even with forged proxy headers`() {
        for (path in listOf("/api/v1/events", "/api/v1/auth/login", "/ws/events", "/actuator/metrics", "/actuator/health/details")) {
            assertEquals(403, get(path, "wrong-secret"), path)
            assertEquals(403, get(path, null), path)
        }
        val forgedPrefix = HttpRequest.newBuilder(URI("http://localhost:$port/api/v1/events"))
            .header("X-Forwarded-Prefix", "/actuator/health")
            .GET()
            .build()
        assertEquals(403, client.send(forgedPrefix, HttpResponse.BodyHandlers.discarding()).statusCode())
    }

    @Test
    fun `edge secret preserves existing security and exact health probe`() {
        assertEquals(200, get("/actuator/health", null))
        assertEquals(200, get("/api/v1/events", "test-origin-secret-0123456789abcdef"))
        assertEquals(401, get("/api/v1/me/cart", "test-origin-secret-0123456789abcdef"))
    }

    @Test
    fun `edge secret allows preflight and signed webhook to reach their handlers`() {
        val preflightUrl = URI("http://localhost:$port/api/v1/auth/login")
        val directPreflight = HttpRequest.newBuilder(preflightUrl)
            .header("Origin", "https://quickpitik.com")
            .header("Access-Control-Request-Method", "POST")
            .method("OPTIONS", HttpRequest.BodyPublishers.noBody())
            .build()
        assertEquals(403, client.send(directPreflight, HttpResponse.BodyHandlers.discarding()).statusCode())

        val preflight = HttpRequest.newBuilder(preflightUrl)
            .header("Origin", "https://quickpitik.com")
            .header("Access-Control-Request-Method", "POST")
            .header("X-Origin-Secret", "test-origin-secret-0123456789abcdef")
            .method("OPTIONS", HttpRequest.BodyPublishers.noBody())
            .build()
        assertEquals(200, client.send(preflight, HttpResponse.BodyHandlers.discarding()).statusCode())

        val body = "not-json"
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec("test-payment-webhook-secret".toByteArray(), "HmacSHA256"))
        val signature = HexFormat.of().formatHex(mac.doFinal(body.toByteArray()))
        val webhook = HttpRequest.newBuilder(URI("http://localhost:$port/api/v1/payments/webhook/stub"))
            .header("X-Origin-Secret", "test-origin-secret-0123456789abcdef")
            .header("X-QuickPitik-Signature", signature)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build()
        assertEquals(400, client.send(webhook, HttpResponse.BodyHandlers.discarding()).statusCode())
    }

    private fun get(path: String, secret: String?): Int {
        val request = HttpRequest.newBuilder(URI("http://localhost:$port$path"))
            .header("CF-Connecting-IP", "203.0.113.9")
            .header("X-Forwarded-For", "198.51.100.10")
            .header("X-Forwarded-Host", "api.quickpitik.com")
            .apply { secret?.let { header("X-Origin-Secret", it) } }
            .GET()
            .build()
        return client.send(request, HttpResponse.BodyHandlers.discarding()).statusCode()
    }
}
