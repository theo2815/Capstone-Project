package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.entity.Role
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.auth.AuthService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse

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
    properties = ["app.rate-limit.enabled=true"],
)
class SecurityMatrixIntegrationTest : PostgresIntegrationTest() {

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
