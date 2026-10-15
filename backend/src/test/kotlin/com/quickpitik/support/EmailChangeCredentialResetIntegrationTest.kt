package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.auth.GoogleLoginRequest
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.profile.EmailChangeRequest
import com.quickpitik.dto.profile.PasswordChangeRequest
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.auth.AuthService
import com.quickpitik.service.auth.GoogleAuthService
import com.quickpitik.service.auth.PasswordResetService
import com.quickpitik.service.email.EmailService
import com.quickpitik.service.profile.EmailChangeService
import com.quickpitik.service.profile.ProfileService
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.security.oauth2.jwt.Jwt
import org.springframework.security.oauth2.jwt.JwtDecoder
import org.springframework.test.context.bean.override.mockito.MockitoBean
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Instant
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@SpringBootTest(
    webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties = ["app.auth.google.client-id=integration-test-client"],
)
class EmailChangeCredentialResetIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var passwords: PasswordEncoder
    @Autowired private lateinit var emailChange: EmailChangeService
    @Autowired private lateinit var google: GoogleAuthService
    @Autowired private lateinit var passwordReset: PasswordResetService
    @Autowired private lateinit var profile: ProfileService
    @Autowired private lateinit var jwt: JwtTokenProvider
    @Autowired private lateinit var auth: AuthService
    @Autowired private lateinit var json: ObjectMapper
    @MockitoBean private lateinit var googleJwtDecoder: JwtDecoder
    @MockitoBean private lateinit var emailService: EmailService

    @LocalServerPort
    private var port: Int = 0

    private val client = HttpClient.newHttpClient()

    // Security audit 2026-10-14, auth/email-change-token-survives-credential-reset.
    private fun squatter(email: String, password: String) = users.saveAndFlush(User(
        email = email, passwordHash = passwords.encode(password), name = "Squatter", role = Role.RUNNER,
    ))   // emailVerifiedAt stays null: registration is advisory

    private fun mailedEmailChangeToken(): String {
        val captor = ArgumentCaptor.forClass(String::class.java)
        Mockito.verify(emailService, Mockito.atLeastOnce()).sendEmailChangeConfirmation(Mockito.anyString(), captor.capture() ?: "")
        return captor.allValues.last()
    }

    private fun stubGoogle(token: String, sub: String, email: String) {
        Mockito.`when`(googleJwtDecoder.decode(token)).thenReturn(
            Jwt.withTokenValue(token)
                .header("alg", "RS256")
                .subject(sub)
                .claim("email", email)
                .claim("email_verified", true)
                .claim("name", "Inbox Owner")
                .issuedAt(Instant.now())
                .expiresAt(Instant.now().plusSeconds(300))
                .build(),
        )
    }

    private fun put(path: String, body: String, token: String): HttpResponse<String> =
        client.send(
            HttpRequest.newBuilder(URI("http://localhost:$port$path"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer $token")
                .PUT(HttpRequest.BodyPublishers.ofString(body))
                .build(),
            HttpResponse.BodyHandlers.ofString(),
        )

    private fun post(path: String, body: String): HttpResponse<String> =
        client.send(
            HttpRequest.newBuilder(URI("http://localhost:$port$path"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build(),
            HttpResponse.BodyHandlers.ofString(),
        )

    private fun errorCode(response: HttpResponse<String>): String =
        json.readTree(response.body()).path("errors").path(0).path("code").asText()

    @Test
    fun `a password change makes the next email change refresh, then succeed`() {
        val email = "owner-${UUID.randomUUID()}@test.local"
        val user = users.saveAndFlush(User(
            email = email, passwordHash = passwords.encode("Owner-pass-1"), name = "Owner", role = Role.RUNNER,
            emailVerifiedAt = OffsetDateTime.now(),
        ))
        val session = auth.login(LoginRequest(email = email, password = "Owner-pass-1"))
        profile.changePassword(user.id, PasswordChangeRequest(
            currentPassword = "Owner-pass-1", newPassword = "Owner-new-pass-2", refreshToken = session.refreshToken,
        ))
        val body = """{"newEmail":"moved-${UUID.randomUUID()}@test.local","currentPassword":"Owner-new-pass-2"}"""

        val stale = put("/api/v1/me/email", body, session.accessToken)
        assertEquals(401, stale.statusCode())
        assertEquals("UNAUTHORIZED", errorCode(stale))

        val refreshed = post("/api/v1/auth/refresh", """{"refreshToken":"${session.refreshToken}"}""")
        assertEquals(200, refreshed.statusCode())
        val freshAccess = json.readTree(refreshed.body()).path("data").path("accessToken").asText()
        val retried = put("/api/v1/me/email", body, freshAccess)

        assertTrue(retried.statusCode() in 200..299, "retry answered ${retried.statusCode()}: ${retried.body()}")
        Mockito.verify(emailService, Mockito.times(1)).sendEmailChangeConfirmation(Mockito.anyString(), Mockito.anyString())
    }

    @Test
    fun `an email-change link minted before the Google takeover guard no longer works`() {
        val email = "owner-${UUID.randomUUID()}@test.local"
        val user = squatter(email, "Squatter-pass-1")
        emailChange.requestChange(user.id, user.credentialVersion, EmailChangeRequest(newEmail = "attacker-${UUID.randomUUID()}@test.local", currentPassword = "Squatter-pass-1"))
        val link = mailedEmailChangeToken()
        stubGoogle("owner-google-token", sub = "sub-${UUID.randomUUID()}", email = email)
        google.login(GoogleLoginRequest(idToken = "owner-google-token"))

        assertFailsWith<ValidationException> { emailChange.confirmChange(link) }
        assertEquals(email, users.findById(user.id).get().email)
    }

    @Test
    fun `an email-change link minted before an OTP password reset no longer works`() {
        val email = "owner-${UUID.randomUUID()}@test.local"
        val user = squatter(email, "Squatter-pass-1")
        emailChange.requestChange(user.id, user.credentialVersion, EmailChangeRequest(newEmail = "attacker-${UUID.randomUUID()}@test.local", currentPassword = "Squatter-pass-1"))
        val link = mailedEmailChangeToken()
        passwordReset.requestReset(email)
        val code = ArgumentCaptor.forClass(String::class.java)
        Mockito.verify(emailService).sendPasswordResetEmail(Mockito.eq(email) ?: email, code.capture() ?: "")
        val cont = passwordReset.verifyOtp(email, code.value)
        passwordReset.confirmReset(cont, "Owner-new-pass-1")

        assertFailsWith<ValidationException> { emailChange.confirmChange(link) }
        assertEquals(email, users.findById(user.id).get().email)
    }

    @Test
    fun `an email-change link minted before a password change no longer works`() {
        val email = "owner-${UUID.randomUUID()}@test.local"
        val user = squatter(email, "Squatter-pass-1")
        emailChange.requestChange(user.id, user.credentialVersion, EmailChangeRequest(newEmail = "attacker-${UUID.randomUUID()}@test.local", currentPassword = "Squatter-pass-1"))
        val link = mailedEmailChangeToken()
        profile.changePassword(user.id, PasswordChangeRequest(currentPassword = "Squatter-pass-1", newPassword = "Owner-new-pass-1"))

        assertFailsWith<ValidationException> { emailChange.confirmChange(link) }
        assertEquals(email, users.findById(user.id).get().email)
    }

    @Test
    fun `the earlier registrant's access token cannot start an email change after the Google takeover`() {
        val email = "owner-${UUID.randomUUID()}@test.local"
        val user = squatter(email, "Squatter-pass-1")
        val staleAccess = jwt.createAccessToken(user)
        stubGoogle("owner-google-token", sub = "sub-${UUID.randomUUID()}", email = email)
        google.login(GoogleLoginRequest(idToken = "owner-google-token")) // guard runs; account is now Google-linked
        Mockito.clearInvocations(emailService)

        val response = put("/api/v1/me/email", """{"newEmail":"attacker-${UUID.randomUUID()}@test.local"}""", staleAccess)

        assertEquals(401, response.statusCode())
        assertEquals("UNAUTHORIZED", errorCode(response))
        Mockito.verify(emailService, Mockito.never()).sendEmailChangeConfirmation(Mockito.anyString(), Mockito.anyString())
    }
}
