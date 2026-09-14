package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.entity.AdminMfa
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.AdminMfaRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.MfaSecretCipher
import com.quickpitik.service.admin.TotpService
import com.quickpitik.service.auth.AuthService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertTimeoutPreemptively
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.security.authentication.BadCredentialsException
import org.springframework.security.crypto.password.PasswordEncoder
import java.time.Duration
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/**
 * The half of the V29 lockout that a mock cannot see.
 *
 * `AuthService.login` throws on a bad password, and the class is
 * `@Transactional` — so the transaction rolls back, taking any write made
 * inside it. `LoginAttemptService.recordFailure` is therefore a separate bean
 * annotated `REQUIRES_NEW`, which commits in a transaction of its own.
 *
 * `AuthServiceTest` verifies that `recordFailure` is *called*. Mockito records
 * that call whether or not the write survives, so a regression that moved the
 * counter back inside `AuthService` — or that dropped `REQUIRES_NEW` — would
 * leave the unit suite fully green while the feature silently stopped working:
 * the counter would never advance past 1 and nothing would ever lock. Only a
 * real transaction shows the difference, which is what this class is for.
 */
class AuthLockoutIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var authService: AuthService

    @Autowired
    private lateinit var userRepository: UserRepository

    @Autowired
    private lateinit var adminMfaRepository: AdminMfaRepository

    @Autowired
    private lateinit var mfaSecretCipher: MfaSecretCipher

    @Autowired
    private lateinit var totpService: TotpService

    @Autowired
    private lateinit var passwordEncoder: PasswordEncoder

    private lateinit var email: String
    private val password = "correct-horse-battery-staple-42"

    @BeforeEach
    fun registerAccount() {
        email = "lockout-${UUID.randomUUID()}@test.local"
        authService.register(
            RegisterRequest(name = "Lockout Test", email = email, password = password, role = Role.RUNNER),
        )
    }

    /** THE test. Everything else here is a consequence of this one holding. */
    @Test
    fun `a failed attempt is persisted even though the login transaction rolls back`() {
        attemptWrongPassword()

        assertEquals(1, reload().failedLoginAttempts)
    }

    @Test
    fun `the counter accumulates across attempts`() {
        repeat(3) { attemptWrongPassword() }

        assertEquals(3, reload().failedLoginAttempts)
    }

    @Test
    fun `the fifth failure locks the account`() {
        repeat(5) { attemptWrongPassword() }

        val user = reload()
        assertNotNull(user.lockedUntil, "account should be locked after 5 failures")
        // The counter resets as the lock goes on, so an expired lock doesn't
        // re-lock on the very next mistake.
        assertEquals(0, user.failedLoginAttempts)
    }

    /** The point of the feature: guessing right on attempt six still gets in nowhere. */
    @Test
    fun `a locked account refuses the correct password`() {
        repeat(5) { attemptWrongPassword() }

        val ex = assertFailsWith<ApiException> {
            authService.login(LoginRequest(email = email, password = password))
        }

        assertEquals(ErrorCodes.ACCOUNT_LOCKED, ex.code)
        assertNotNull(ex.retryAfterSeconds)
    }

    @Test
    fun `a good password before the threshold clears the streak`() {
        repeat(3) { attemptWrongPassword() }

        authService.login(LoginRequest(email = email, password = password))

        val user = reload()
        assertEquals(0, user.failedLoginAttempts)
        assertNull(user.lockedUntil)
    }

    @Test
    fun `an elapsed lock lets the user back in`() {
        repeat(5) { attemptWrongPassword() }
        expireTheLock()

        authService.login(LoginRequest(email = email, password = password))

        assertNull(reload().lockedUntil, "a successful login should clear the stale lock")
    }

    @Test
    fun `MFA failures persist without deadlocking and a new challenge does not clear them`() {
        assertTimeoutPreemptively(Duration.ofSeconds(20)) {
            val adminEmail = "mfa-lockout-${UUID.randomUUID()}@test.local"
            val admin = userRepository.saveAndFlush(
                User(
                    email = adminEmail,
                    passwordHash = passwordEncoder.encode(password),
                    name = "MFA Lockout Test",
                    role = Role.ADMIN,
                    emailVerifiedAt = OffsetDateTime.now(),
                ),
            )
            val secret = totpService.randomSecret()
            adminMfaRepository.saveAndFlush(
                AdminMfa(userId = admin.id, secretEnc = mfaSecretCipher.encrypt(secret), enabled = true),
            )
            var challenge = requireNotNull(
                authService.login(LoginRequest(email = adminEmail, password = password)).mfaToken,
            )

            repeat(4) {
                val ex = assertFailsWith<ApiException> { authService.verifyMfa(challenge, "wrong-code") }
                assertEquals(ErrorCodes.INVALID_MFA_CODE, ex.code)
            }
            assertEquals(4, requireNotNull(userRepository.findByEmail(adminEmail)).failedLoginAttempts)

            challenge = requireNotNull(
                authService.login(LoginRequest(email = adminEmail, password = password)).mfaToken,
            )
            assertEquals(4, requireNotNull(userRepository.findByEmail(adminEmail)).failedLoginAttempts)

            val fifth = assertFailsWith<ApiException> { authService.verifyMfa(challenge, "wrong-code") }
            assertEquals(ErrorCodes.INVALID_MFA_CODE, fifth.code)
            assertNotNull(requireNotNull(userRepository.findByEmail(adminEmail)).lockedUntil)

            val sixth = assertFailsWith<ApiException> { authService.verifyMfa(challenge, "wrong-code") }
            assertEquals(ErrorCodes.ACCOUNT_LOCKED, sixth.code)
        }
    }

    // ─── helpers ──────────────────────────────────────────────────────────

    private fun attemptWrongPassword() {
        assertFailsWith<BadCredentialsException> {
            authService.login(LoginRequest(email = email, password = "definitely-not-the-password"))
        }
    }

    /** Backdates the lock rather than sleeping 15 minutes. */
    private fun expireTheLock() {
        val user = reload()
        user.lockedUntil = OffsetDateTime.now().minusMinutes(1)
        userRepository.saveAndFlush(user)
    }

    private fun reload() = requireNotNull(userRepository.findByEmail(email)) { "test user vanished" }
}
