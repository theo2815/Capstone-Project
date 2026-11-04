package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.LegalProperties
import com.quickpitik.dto.auth.GoogleLoginRequest
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.dto.profile.PasswordChangeRequest
import com.quickpitik.entity.Role
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.auth.AuthService
import com.quickpitik.service.email.EmailService
import com.quickpitik.service.auth.GoogleAuthService
import com.quickpitik.service.auth.PasswordResetService
import com.quickpitik.service.profile.ProfileService
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.security.oauth2.jwt.Jwt
import org.springframework.security.oauth2.jwt.JwtDecoder
import org.springframework.test.context.TestPropertySource
import org.springframework.test.context.bean.override.mockito.MockitoBean
import java.time.Instant
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull

@TestPropertySource(properties = ["app.auth.google.client-id=integration-test-client"])
class GoogleAccountIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var googleAuth: GoogleAuthService
    @Autowired private lateinit var auth: AuthService
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var passwords: PasswordResetService
    @Autowired private lateinit var profile: ProfileService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var legal: LegalProperties
    @MockitoBean private lateinit var googleJwtDecoder: JwtDecoder
    @MockitoBean private lateinit var emailService: EmailService

    @Test
    fun `Google account cannot be registered again with the same normalized email and can set a password with OTP`() {
        val email = "google-${UUID.randomUUID()}@test.local"
        val sub = UUID.randomUUID().toString()
        Mockito.`when`(googleJwtDecoder.decode("verified-google-token")).thenReturn(
            Jwt.withTokenValue("verified-google-token")
                .header("alg", "RS256")
                .subject(sub)
                .claim("email", email.uppercase())
                .claim("email_verified", true)
                .claim("name", "Google Runner")
                .issuedAt(Instant.now())
                .expiresAt(Instant.now().plusSeconds(300))
                .build(),
        )

        val account = googleAuth.login(GoogleLoginRequest("verified-google-token", Role.RUNNER)).user
        assertEquals(account.id, googleAuth.login(GoogleLoginRequest("verified-google-token")).user.id)
        for (duplicate in listOf(email, "  ${email.uppercase()}  ")) {
            val error = assertFailsWith<ConflictException> {
                auth.register(RegisterRequest("Duplicate", duplicate, "marathon-photos-2026", Role.PHOTOGRAPHER))
            }
            assertEquals(ErrorCodes.EMAIL_TAKEN, error.code)
        }
        assertEquals(account.id, assertNotNull(users.findByEmail(email)).id)
        assertEquals(Role.RUNNER, assertNotNull(users.findByGoogleSub(sub)).role)

        passwords.requestReset(email)
        val code = ArgumentCaptor.forClass(String::class.java)
        Mockito.verify(emailService).sendPasswordResetEmail(Mockito.eq(email) ?: email, code.capture() ?: "")
        val continuation = passwords.verifyOtp(email, code.value)
        val newPassword = "verified-marathon-password-2026"
        passwords.confirmReset(continuation, newPassword)
        assertEquals(account.id, auth.login(LoginRequest(email, newPassword)).user.id)
        assertEquals(account.id, googleAuth.login(GoogleLoginRequest("verified-google-token")).user.id)
        assertFailsWith<UnauthorizedException> {
            profile.changePassword(account.id, PasswordChangeRequest(null, "session-only-replacement-2026"))
        }
    }

    // Proves V91's columns and the entity mapping against real Postgres.
    @Test
    fun `new Google account with termsAccepted stores the configured terms version`() {
        val email = "terms-${UUID.randomUUID()}@test.local"
        Mockito.`when`(googleJwtDecoder.decode("terms-token")).thenReturn(
            Jwt.withTokenValue("terms-token")
                .header("alg", "RS256")
                .subject(UUID.randomUUID().toString())
                .claim("email", email)
                .claim("email_verified", true)
                .issuedAt(Instant.now())
                .expiresAt(Instant.now().plusSeconds(300))
                .build(),
        )

        val id = googleAuth.login(GoogleLoginRequest("terms-token", Role.RUNNER, termsAccepted = true)).user.id

        val row = jdbc.queryForMap("SELECT terms_version, terms_accepted_at FROM users WHERE id = ?", id)
        assertEquals(legal.currentVersion, row["terms_version"])
        assertNotNull(row["terms_accepted_at"])
    }
}
