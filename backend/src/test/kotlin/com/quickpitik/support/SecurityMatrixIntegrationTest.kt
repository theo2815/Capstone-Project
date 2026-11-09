package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.PaymentWebhookProperties
import com.quickpitik.config.PaymongoProperties
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.entity.Role
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.auth.AuthService
import com.quickpitik.service.auth.DesktopAuthService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Instant
import java.nio.file.Files
import java.nio.file.Path
import java.util.HexFormat
import java.util.UUID
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The route matrix in `SecurityConfig` and the `@PreAuthorize` role gates on the
 * controllers had never been evaluated by a test: every controller test builds
 * the controller by hand, and `SecurityConfigGuestOrderTest` only exercises the
 * guest-order regex factories (2026-09-11 test-suite audit). This class drives
 * the real filter chain over HTTP — one row per rule that would be a hole if it
 * silently moved — so a misplaced `permitAll` or a dropped `hasRole` fails CI.
 *
 * Real HTTP rather than MockMvc for the same reason as
 * `PhotoBundleHttpIntegrationTest`: MockMvc's request shape differs from a
 * browser's in ways that have already produced a false green once.
 */
@SpringBootTest(
    webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties = ["app.rate-limit.enabled=true", "app.ai-api.webhook.hmac-secret=test-only-ai-webhook-secret"],
)
class SecurityMatrixIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var aiApiProperties: AiApiProperties

    @Autowired
    private lateinit var paymentWebhookProperties: PaymentWebhookProperties

    @Autowired
    private lateinit var paymongoProperties: PaymongoProperties

    @Autowired
    private lateinit var authService: AuthService

    @Autowired
    private lateinit var jwtTokenProvider: JwtTokenProvider

    @LocalServerPort
    private var port: Int = 0

    @Value("\${app.admin.bootstrap-email}")
    private lateinit var adminEmail: String

    @Value("\${app.admin.bootstrap-password}")
    private lateinit var adminPassword: String

    private val client = HttpClient.newHttpClient()
    private val json = ObjectMapper()

    @Test
    fun `anonymous requests reach exactly the permitted routes`() {
        assertEquals(200, get("/actuator/health").statusCode())
        assertEquals(200, get("/api/v1/events").statusCode())
        assertEquals(200, get("/api/v1/regions").statusCode())
        assertEquals(200, get("/api/v1/race-types").statusCode())
        // Guest checkout is permitAll: an empty body reaches the controller and
        // fails validation — it is not bounced by the filter chain.
        assertEquals(400, post("/api/v1/orders", "{}").statusCode())

        assertEquals(401, get("/actuator/metrics").statusCode())
        assertEquals(401, get("/api/v1/me/cart").statusCode())
        // Deliberately NOT in the permitAll list next to its /auth siblings.
        assertEquals(401, post("/api/v1/auth/resend-verification", "").statusCode())
        val me = get("/api/v1/auth/me")
        assertEquals(401, me.statusCode())
        assertEquals("UNAUTHORIZED", errorCode(me))
    }

    @Test
    fun `edge routes answer as their own guards say`() {
        // The public events rule lists its handler shapes; the RUNNER-only
        // photo-alert status under the same prefix is not one of them.
        assertEquals(401, get("/api/v1/events/any-slug/photo-alert").statusCode())
        // The mobile bridge page embeds an order capability token: never cached.
        val bridge = get("/api/v1/orders/mobile-return?orderId=o1&token=t1")
        assertEquals(200, bridge.statusCode())
        assertTrue("no-store" in bridge.headers().firstValue("Cache-Control").orElse(""))
        // Webhooks verify the signature before parsing: unsigned junk is a 401, not a 400.
        assertEquals(401, post("/api/v1/payments/webhook/stub", "not-json").statusCode())
        assertEquals(401, post("/api/v1/internal/ai-webhooks", "not-json").statusCode())
        assertEquals(401, post("/api/v1/payments/webhook/paymongo", "not-json").statusCode())
        assertEquals(401, post("/api/v1/payments/webhook/paymongo-transfers", "not-json").statusCode())
        // An absent body must still reach the signature verifier.
        assertEquals(401, post("/api/v1/payments/webhook/stub", "").statusCode())
        assertEquals(401, post("/api/v1/internal/ai-webhooks", "").statusCode())
        assertEquals(401, post("/api/v1/payments/webhook/paymongo", "").statusCode())
        assertEquals(401, post("/api/v1/payments/webhook/paymongo-transfers", "").statusCode())
    }

    @Test
    fun `signed malformed webhook bodies return validation errors on all four routes`() {
        val body = "not-json"
        val paymongoTimestamp = Instant.now().epochSecond.toString()
        val paymongoSignature = "t=$paymongoTimestamp,te=${hmac(paymongoProperties.webhookSecret, "$paymongoTimestamp.$body")}"
        val routes = listOf(
            Triple("/api/v1/payments/webhook/stub", paymentWebhookProperties.signatureHeader, hmac(paymentWebhookProperties.hmacSecret, body)),
            Triple("/api/v1/internal/ai-webhooks", "X-QuickPitik-Signature", "sha256=${hmac(aiApiProperties.webhook.hmacSecret, body)}"),
            Triple("/api/v1/payments/webhook/paymongo", "Paymongo-Signature", paymongoSignature),
            Triple("/api/v1/payments/webhook/paymongo-transfers", "Paymongo-Signature", paymongoSignature),
        )

        routes.forEach { (path, header, signature) ->
            val response = client.send(
                HttpRequest.newBuilder(uri(path))
                    .header("Content-Type", "application/json")
                    .header(header, signature)
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build(),
                HttpResponse.BodyHandlers.ofString(),
            )
            assertEquals(400, response.statusCode(), path)
            assertEquals("VALIDATION_ERROR", errorCode(response), path)
            assertFalse(response.body().contains(body), path)
        }
    }

    private fun hmac(secret: String, payload: String): String {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(secret.toByteArray(Charsets.UTF_8), "HmacSHA256"))
        return HexFormat.of().formatHex(mac.doFinal(payload.toByteArray(Charsets.UTF_8)))
    }

    @Test
    fun `role gates hold end to end`() {
        val runner = register(Role.RUNNER)
        val photographer = register(Role.PHOTOGRAPHER)
        val admin = adminToken()

        assertEquals(200, get("/api/v1/me/cart", runner).statusCode())
        assertEquals(403, get("/api/v1/me/cart", photographer).statusCode())
        assertEquals(403, get("/api/v1/me/photographer/events", runner).statusCode())
        val forbidden = get("/api/v1/admin/users", runner)
        assertEquals(403, forbidden.statusCode())
        assertEquals("FORBIDDEN", errorCode(forbidden))
        assertEquals(200, get("/api/v1/admin/users", admin).statusCode())
        assertEquals(200, get("/actuator/metrics", admin).statusCode())
    }

    @Test
    fun `admin command lookup is admin only and hides unknown keys`() {
        val path = "/api/v1/admin/commands/${UUID.randomUUID()}"
        assertEquals(401, get(path).statusCode())
        assertEquals(403, get(path, register(Role.RUNNER)).statusCode())
        assertEquals(403, get(path, register(Role.PHOTOGRAPHER)).statusCode())
        val admin = adminToken()
        val missing = get(path, admin)
        assertEquals(404, missing.statusCode())
        // Not the generic NOT_FOUND an unmapped path answers: the website releases a stored key only on this code.
        assertEquals("ADMIN_COMMAND_NOT_FOUND", errorCode(missing))
        assertEquals(400, get("/api/v1/admin/commands/not-a-uuid", admin).statusCode())
    }

    @Test
    fun `private removed event history keeps photographer role gate`() {
        val path = "/api/v1/me/photographer/events/${UUID.randomUUID()}?includeRemoved=true"
        assertEquals(401, get(path).statusCode())
        assertEquals(403, get(path, register(Role.RUNNER)).statusCode())
        assertEquals(404, get(path, register(Role.PHOTOGRAPHER)).statusCode())
    }

    @Test
    fun `metadata routes obey photographer and anonymous gates with no store`() {
        val id = UUID.randomUUID()
        val ownerPath = "/api/v1/me/photographer/photos/$id/metadata"
        val publicPath = "/api/v1/public/photos/$id/metadata"
        assertEquals(401, get(ownerPath).statusCode())
        assertEquals(403, get(ownerPath, register(Role.RUNNER)).statusCode())
        val photographer = register(Role.PHOTOGRAPHER)
        val settings = get("/api/v1/me/photographer/metadata-settings", photographer)
        assertEquals(200, settings.statusCode())
        assertTrue("no-store" in settings.headers().firstValue("Cache-Control").orElse(""))
        val missingOwner = get(ownerPath, photographer)
        assertEquals(404, missingOwner.statusCode())
        assertTrue("no-store" in missingOwner.headers().firstValue("Cache-Control").orElse(""))
        val missingPublic = get(publicPath)
        assertEquals(404, missingPublic.statusCode())
        assertTrue("no-store" in missingPublic.headers().firstValue("Cache-Control").orElse(""))
    }

    @Test
    fun `direct upload controls require photographer auth and are never cached`() {
        val eventId = UUID.randomUUID()
        val photoId = UUID.randomUUID()
        val beginBody = """{"contentHash":"${"a".repeat(64)}","contentType":"image/jpeg","sizeBytes":123}"""
        val commitBody = """{"photoId":"$photoId","key":"staging/photo-uploads/x","contentHash":"${"a".repeat(64)}","contentType":"image/jpeg","sizeBytes":123}"""

        assertEquals(401, post("/api/v1/me/photographer/events/$eventId/photos/direct", beginBody).statusCode())
        assertEquals(
            403,
            post("/api/v1/me/photographer/events/$eventId/photos/direct", beginBody, register(Role.RUNNER)).statusCode(),
        )

        val photographer = register(Role.PHOTOGRAPHER)
        val begin = post("/api/v1/me/photographer/events/$eventId/photos/direct", beginBody, photographer)
        val commit = post("/api/v1/me/photographer/events/$eventId/photos/direct/commit", commitBody, photographer)
        assertEquals(404, begin.statusCode())
        assertEquals(403, commit.statusCode())
        assertTrue("no-store" in begin.headers().firstValue("Cache-Control").orElse(""))
        assertTrue("no-store" in commit.headers().firstValue("Cache-Control").orElse(""))
    }

    @Test
    fun `desktop connect mints for photographers only and redeems a code once`() {
        val verifier = "v".repeat(43)
        val challenge = DesktopAuthService.challengeOf(verifier)
        val mintBody = """{"codeChallenge":"$challenge"}"""

        assertEquals(401, post("/api/v1/auth/desktop/codes", mintBody).statusCode())
        assertEquals(403, post("/api/v1/auth/desktop/codes", mintBody, register(Role.RUNNER)).statusCode())

        val minted = post("/api/v1/auth/desktop/codes", mintBody, register(Role.PHOTOGRAPHER))
        assertEquals(200, minted.statusCode())
        assertTrue("no-store" in minted.headers().firstValue("Cache-Control").orElse(""))
        val code = json.readTree(minted.body()).path("data").path("code").asText()
        assertEquals(43, code.length)

        // Anonymous and permitAll: a bogus code reaches the service and is refused
        // there (INVALID_TOKEN), not by the filter chain (UNAUTHORIZED).
        val bogus = post("/api/v1/auth/desktop/exchange", """{"code":"${"x".repeat(43)}","codeVerifier":"$verifier"}""")
        assertEquals(401, bogus.statusCode())
        assertEquals("INVALID_TOKEN", errorCode(bogus))

        val exchangeBody = """{"code":"$code","codeVerifier":"$verifier"}"""
        val exchanged = post("/api/v1/auth/desktop/exchange", exchangeBody)
        assertEquals(200, exchanged.statusCode())
        assertTrue("no-store" in exchanged.headers().firstValue("Cache-Control").orElse(""))
        assertTrue(json.readTree(exchanged.body()).path("data").path("accessToken").asText().isNotBlank())
        assertTrue(json.readTree(exchanged.body()).path("data").path("refreshToken").asText().isNotBlank())

        val replay = post("/api/v1/auth/desktop/exchange", exchangeBody)
        assertEquals(401, replay.statusCode())
        assertEquals("INVALID_TOKEN", errorCode(replay))
    }

    @Test
    fun `desktop code exchange shares the login rate limit`() {
        val body = """{"code":"${"x".repeat(43)}","codeVerifier":"${"v".repeat(43)}"}"""
        val responses = (1..11).map {
            val request = HttpRequest.newBuilder(uri("/api/v1/auth/desktop/exchange"))
                .header("Content-Type", "application/json")
                .header("CF-Connecting-IP", "203.0.113.10")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build()
            client.send(request, HttpResponse.BodyHandlers.ofString())
        }

        assertEquals(401, responses.first().statusCode())
        assertEquals(429, responses.last().statusCode())
    }

    @Test
    fun `first-factor and step-up tokens are refused as bearers over HTTP`() {
        val userId = UUID.randomUUID()

        assertEquals(401, get("/api/v1/auth/me", jwtTokenProvider.createMfaPendingToken(userId)).statusCode())
        assertEquals(401, get("/api/v1/auth/me", jwtTokenProvider.createStepUpToken(userId)).statusCode())
    }

    @Test
    fun `auth rate limit uses the Cloudflare client IP when X-Forwarded-For changes`() {
        val responses = (1..11).map { i ->
            val request = HttpRequest.newBuilder(uri("/api/v1/auth/login"))
                .header("Content-Type", "application/json")
                .header("CF-Connecting-IP", "203.0.113.9")
                .header("X-Forwarded-For", "198.51.100.$i")
                .POST(HttpRequest.BodyPublishers.ofString("""{"email":"missing@test.local","password":"wrong-password"}"""))
                .build()
            client.send(request, HttpResponse.BodyHandlers.ofString())
        }

        assertEquals(429, responses.last().statusCode())
    }

    private fun register(role: Role): String = authService.register(
        RegisterRequest(
            name = "Matrix ${role.name}",
            email = "matrix-${UUID.randomUUID()}@test.local",
            password = "correct-horse-battery-staple-42",
            role = role,
        ),
    ).accessToken

    @Test
    fun `case evidence and its known local storage key stay private`() {
        val id = UUID.randomUUID()
        val route = "/api/v1/admin/photo-removal-requests/$id/evidence"
        assertEquals(401, get(route).statusCode())
        assertEquals(403, get(route, register(Role.RUNNER)).statusCode())
        assertEquals(403, get(route, register(Role.PHOTOGRAPHER)).statusCode())
        val key = "photo-removal/$id/evidence/${UUID.randomUUID()}"
        val path = Path.of("build/test-storage", key)
        Files.createDirectories(path.parent)
        Files.writeString(path, "synthetic private evidence")
        try {
            assertTrue(get("/storage/$key").statusCode() in setOf(401, 403, 404))
            assertTrue(get("/storage/$key", adminToken()).statusCode() in setOf(401, 403, 404))
        } finally {
            Files.deleteIfExists(path)
        }
    }

    @Test
    fun `known case evidence storage prefix cannot bypass the private stream`() {
        val key = "photo-removal/${UUID.randomUUID()}/evidence/${UUID.randomUUID()}"
        val path = Path.of("build/test-storage", key)
        Files.createDirectories(path.parent)
        Files.writeString(path, "synthetic private evidence")
        try {
            assertTrue(get("/storage/$key").statusCode() in setOf(401, 403, 404))
            assertTrue(get("/storage/$key", adminToken()).statusCode() in setOf(401, 403, 404))
        } finally {
            Files.deleteIfExists(path)
        }
    }

    @Test
    fun `removal intake shares the bounded admin media write policy`() {
        val token = adminToken()
        val responses = (1..21).map {
            val body = """{"eventId":"${UUID.randomUUID()}","supportReference":"synthetic-ticket","requesterContact":"synthetic-contact","reason":"privacy","commandId":"${UUID.randomUUID()}"}"""
            post("/api/v1/admin/photo-removal-requests", body, token)
        }
        assertEquals(404, responses.first().statusCode())
        assertEquals(429, responses.last().statusCode())
        assertTrue(responses.last().headers().firstValue("Retry-After").isPresent)
    }

    @Test
    fun `removal gallery browsing is not bound by the face-search policy`() {
        val token = adminToken()
        val id = UUID.randomUUID()
        val gallery = (1..31).map { page -> get("/api/v1/admin/photo-removal-requests/$id/photos?offset=${page * 24}&limit=24", token) }
        assertTrue(gallery.none { it.statusCode() == 429 }, "gallery paging must not drain the 30-per-15-minute search bucket")
        assertEquals(404, gallery.last().statusCode())
        val body = """{"commandId":"${UUID.randomUUID()}","expectedVersion":0}"""
        val matches = (1..31).map { post("/api/v1/admin/photo-removal-requests/$id/matches", body, token) }
        assertEquals(429, matches.last().statusCode())
        assertTrue(matches.last().headers().firstValue("Retry-After").isPresent)
    }

    @Test
    fun `mixed case local evidence prefix cannot reveal a known physical key`() {
        val key = "photo-removal/${UUID.randomUUID()}/evidence/${UUID.randomUUID()}"
        val path = Path.of("build/test-storage", key)
        Files.createDirectories(path.parent)
        Files.writeString(path, "synthetic private evidence")
        try {
            assertTrue(get("/storage/$key").statusCode() in setOf(401, 403, 404))
            val mixedCase = "/storage/PHOTO-REMOVAL/${key.substringAfter('/')}"
            assertTrue(get(mixedCase).statusCode() in setOf(401, 403, 404))
            assertTrue(get(mixedCase, adminToken()).statusCode() in setOf(401, 403, 404))
        } finally { Files.deleteIfExists(path) }
    }

    @Test
    fun `photo removal owner routes enforce photographer role and private cache`() {
        val root = "/api/v1/me/photographer/photo-removal-requests"
        val runner = register(Role.RUNNER)
        val admin = adminToken()
        val photographer = register(Role.PHOTOGRAPHER)
        assertEquals(200, get(root, photographer).statusCode())
        val id = UUID.randomUUID()
        val body = """{"commandId":"${UUID.randomUUID()}","expectedVersion":0,"itemIds":["${UUID.randomUUID()}"],"response":"AGREE"}"""
        for (path in listOf(root, "$root/$id")) {
            assertEquals(401, get(path).statusCode())
            assertEquals(403, get(path, runner).statusCode())
            assertEquals(403, get(path, admin).statusCode())
        }
        assertEquals(401, post("$root/$id/responses", body).statusCode())
        assertEquals(403, post("$root/$id/responses", body, runner).statusCode())
        assertEquals(403, post("$root/$id/responses", body, admin).statusCode())
        val missing = get("$root/$id", photographer)
        assertEquals(404, missing.statusCode())
        assertTrue("no-store" in missing.headers().firstValue("Cache-Control").orElse(""))
        assertEquals(404, post("$root/$id/responses", body, photographer).statusCode())
    }

    @Test
    fun `seller details routes are photographer only and private to their owner`() {
        val path = "/api/v1/me/photographer/seller-details"
        val form = multipart(mapOf("legalName" to "Juan Dela Cruz"))
        assertEquals(401, get(path).statusCode())
        assertEquals(401, form(path, null).statusCode())
        for (token in listOf(register(Role.RUNNER), adminToken())) {
            assertEquals(403, get(path, token).statusCode())
            assertEquals(403, form(path, token).statusCode())
        }

        val (ownerToken, ownerId) = registerPhotographerWithId()
        val other = register(Role.PHOTOGRAPHER)
        jdbc.update(
            "INSERT INTO photographer_seller_details (user_id, legal_name, id_type, id_front_key, selfie_key, address, phone, phone_key, status, revision, submitted_at) " +
                "VALUES (?, 'Juan Dela Cruz', 'PASSPORT', ?, ?, 'Cebu', '09171234567', '9171234567', 'MANUAL_REVIEW', 1, now())",
            ownerId, "seller-details/$ownerId/${UUID.randomUUID()}.jpg", "seller-details/$ownerId/${UUID.randomUUID()}.jpg",
        )
        val mine = get(path, ownerToken)
        assertEquals(200, mine.statusCode())
        assertTrue("no-store" in mine.headers().firstValue("Cache-Control").orElse(""))
        assertEquals("Juan Dela Cruz", json.readTree(mine.body()).path("data").path("legalName").asText())
        // Old clients keep reading "submitted"; new clients read the detailed reviewState.
        assertEquals("submitted", json.readTree(mine.body()).path("data").path("status").asText())
        assertEquals("manual_review", json.readTree(mine.body()).path("data").path("reviewState").asText())
        assertFalse("seller-details/" in mine.body(), "E1 must never carry a storage key")
        val theirs = get(path, other)
        assertEquals(200, theirs.statusCode())
        assertEquals("missing", json.readTree(theirs.body()).path("data").path("status").asText())
        assertFalse("Juan" in theirs.body())

        // A photographer's incomplete form reaches the service and names the missing part.
        val incomplete = form(path, other)
        assertEquals(400, incomplete.statusCode())
        assertEquals("VALIDATION_ERROR", errorCode(incomplete))
    }

    @Test
    fun `the live selfie session route is photographer only, needs consent and answers 503 with AI off`() {
        val path = "/api/v1/me/photographer/seller-details/liveness-session"
        val consent = """{"biometricConsent":true}"""
        assertEquals(401, post(path, consent).statusCode())
        assertEquals(403, post(path, consent, register(Role.RUNNER)).statusCode())

        val photographer = register(Role.PHOTOGRAPHER)
        for (body in listOf("""{"biometricConsent":false}""", "{}")) {
            val refused = post(path, body, photographer)
            assertEquals(400, refused.statusCode(), body)
            assertEquals("VALIDATION_ERROR", errorCode(refused))
            assertEquals("biometricConsent", json.readTree(refused.body()).path("errors").path(0).path("field").asText())
        }
        // The test profile runs AI off: the client offers "Have an admin check instead".
        val off = post(path, consent, photographer)
        assertEquals(503, off.statusCode())
        assertEquals("LIVENESS_UNAVAILABLE", errorCode(off))
        assertTrue("no-store" in off.headers().firstValue("Cache-Control").orElse(""))
    }

    @Test
    fun `the live check route is photographer only, needs a manual review record, and shows no score or reason`() {
        val path = "/api/v1/me/photographer/seller-details/live-check"
        val session = "lv-${UUID.randomUUID()}"
        val body = """{"livenessSessionId":"$session"}"""
        assertEquals(401, post(path, body).statusCode())
        assertEquals(403, post(path, body, register(Role.RUNNER)).statusCode())

        val (token, userId) = registerPhotographerWithId()
        val noRecord = post(path, body, token)
        assertEquals(409, noRecord.statusCode())
        assertEquals("INVALID_STATE_TRANSITION", errorCode(noRecord))
        assertEquals(400, post(path, """{"livenessSessionId":" "}""", token).statusCode())

        jdbc.update(
            "INSERT INTO photographer_seller_details (user_id, legal_name, id_type, id_front_key, selfie_key, address, phone, phone_key, status, revision, submitted_at) " +
                "VALUES (?, 'Juan Dela Cruz', 'PASSPORT', ?, ?, 'Cebu', '09171234567', '9171234567', 'MANUAL_REVIEW', 1, now())",
            userId, "seller-details/$userId/${UUID.randomUUID()}.jpg", "seller-details/$userId/${UUID.randomUUID()}.jpg",
        )
        jdbc.update(
            "INSERT INTO liveness_sessions (session_id, user_id, purpose, consent_version, created_at) VALUES (?, ?, 'PHOTOGRAPHER_IDENTITY', '1.2', now())",
            session, userId,
        )
        // The test profile runs AI off: the session is used up and the record stays with an admin.
        val checked = post(path, body, token)
        assertEquals(200, checked.statusCode(), checked.body())
        val data = json.readTree(checked.body()).path("data")
        assertEquals("manual_review", data.path("reviewState").asText())
        assertEquals("submitted", data.path("status").asText())
        val fields = data.fieldNames().asSequence().map { it.lowercase() }.toList()
        assertTrue(fields.none { "reason" in it && it != "rejectionreason" || "score" in it || "confidence" in it || "similarity" in it }, fields.toString())
        assertEquals("[\"AI_DISABLED\"]", jdbc.queryForObject(
            "SELECT reason_codes::text FROM photographer_seller_details WHERE user_id = ?", String::class.java, userId,
        )?.replace(" ", ""))
        assertEquals(1, jdbc.queryForObject(
            "SELECT count(*) FROM liveness_sessions WHERE session_id = ? AND consumed_at IS NOT NULL", Int::class.java, session,
        ))
        val again = post(path, body, token)
        assertEquals(409, again.statusCode())
        assertEquals("LIVENESS_SESSION_INVALID", errorCode(again))
    }

    @Test
    fun `admin seller details routes are admin only`() {
        val root = "/api/v1/admin/seller-details"
        val (_, sellerId) = registerPhotographerWithId()
        jdbc.update(
            "INSERT INTO photographer_seller_details (user_id, legal_name, id_type, id_front_key, selfie_key, address, phone, phone_key, status, revision, submitted_at) " +
                "VALUES (?, 'Juan Dela Cruz', 'PASSPORT', ?, ?, 'Cebu', '09171234567', '9171234567', 'MANUAL_REVIEW', 1, now())",
            sellerId, "seller-details/$sellerId/${UUID.randomUUID()}.jpg", "seller-details/$sellerId/${UUID.randomUUID()}.jpg",
        )
        val approve = """{"revision":1}"""
        val reject = """{"revision":1,"reason":"blurry"}"""
        for (token in listOf(null, register(Role.RUNNER), register(Role.PHOTOGRAPHER))) {
            val expected = if (token == null) 401 else 403
            assertEquals(expected, get(root, token).statusCode())
            assertEquals(expected, get("$root/$sellerId", token).statusCode())
            assertEquals(expected, post("$root/$sellerId/approve", approve, token).statusCode())
            assertEquals(expected, post("$root/$sellerId/reject", reject, token).statusCode())
        }

        val admin = adminToken()
        val list = get("$root?status=submitted", admin)
        assertEquals(200, list.statusCode())
        assertTrue(sellerId.toString() in list.body())
        assertFalse("seller-details/" in list.body(), "E3 must never carry a storage key")
        val view = get("$root/$sellerId", admin)
        assertEquals(200, view.statusCode())
        assertTrue("no-store" in view.headers().firstValue("Cache-Control").orElse(""))
        assertTrue(json.readTree(view.body()).path("data").path("files").path("idFront").asText().isNotBlank())
        assertEquals(1, jdbc.queryForObject(
            "SELECT count(*) FROM admin_decision_log WHERE target_user_id = ? AND decision = 'seller_details_viewed'", Int::class.java, sellerId,
        ))
        assertEquals(400, post("$root/$sellerId/reject", """{"revision":1,"reason":"  "}""", admin).statusCode())
        assertEquals(404, get("$root/${UUID.randomUUID()}", admin).statusCode())
    }

    @Test
    fun `a known seller-details storage key is never served from local storage`() {
        val key = "seller-details/${UUID.randomUUID()}/${UUID.randomUUID()}.jpg"
        val path = Path.of("build/test-storage", key)
        Files.createDirectories(path.parent)
        Files.writeString(path, "synthetic ID photo")
        try {
            for (token in listOf(null, register(Role.PHOTOGRAPHER), adminToken())) {
                assertTrue(get("/storage/$key", token).statusCode() in setOf(401, 403, 404), "token=$token")
            }
            assertTrue(get("/storage/SELLER-DETAILS/${key.substringAfter('/')}").statusCode() in setOf(401, 403, 404))
        } finally { Files.deleteIfExists(path) }
    }

    @Autowired
    private lateinit var jdbc: org.springframework.jdbc.core.JdbcTemplate

    private fun registerPhotographerWithId(): Pair<String, UUID> {
        val email = "matrix-${UUID.randomUUID()}@test.local"
        val token = authService.register(
            RegisterRequest(name = "Matrix seller", email = email, password = "correct-horse-battery-staple-42", role = Role.PHOTOGRAPHER),
        ).accessToken
        return token to jdbc.queryForObject("SELECT id FROM users WHERE email = ?", UUID::class.java, email)!!
    }

    /** A real multipart body (text parts only): the role check must refuse it, not the binder. */
    private fun multipart(fields: Map<String, String>): (String, String?) -> HttpResponse<String> = { path, token ->
        val boundary = "matrix-${UUID.randomUUID()}"
        val body = fields.entries.joinToString("") { (name, value) ->
            "--$boundary\r\nContent-Disposition: form-data; name=\"$name\"\r\n\r\n$value\r\n"
        } + "--$boundary--\r\n"
        send(
            HttpRequest.newBuilder(uri(path))
                .header("Content-Type", "multipart/form-data; boundary=$boundary")
                .POST(HttpRequest.BodyPublishers.ofString(body)),
            token,
        )
    }

    private fun adminToken(): String {
        val response = authService.login(LoginRequest(email = adminEmail, password = adminPassword))
        assertFalse(response.mfaRequired, "the bootstrap admin has no MFA enrolled")
        return response.accessToken
    }

    private fun get(path: String, token: String? = null): HttpResponse<String> =
        send(HttpRequest.newBuilder(uri(path)).GET(), token)

    private fun post(path: String, body: String, token: String? = null): HttpResponse<String> =
        send(
            HttpRequest.newBuilder(uri(path))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body)),
            token,
        )

    private fun send(builder: HttpRequest.Builder, token: String?): HttpResponse<String> {
        token?.let { builder.header("Authorization", "Bearer $it") }
        return client.send(builder.build(), HttpResponse.BodyHandlers.ofString())
    }

    private fun uri(path: String) = URI("http://localhost:$port$path")

    private fun errorCode(response: HttpResponse<String>): String =
        json.readTree(response.body()).path("errors").path(0).path("code").asText()
}
