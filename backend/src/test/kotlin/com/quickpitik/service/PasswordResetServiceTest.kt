package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.PasswordResetToken
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PasswordResetTokenRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.OpaqueTokens
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.mockito.Mockito
import org.springframework.security.crypto.password.PasswordEncoder
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// Three-step OTP reset flow (V37). The security properties under test:
//
// - Verify is unbypassable: a row is born with tokenHash NULL and only a
//   correct code rotates it into the continuation token confirmReset accepts,
//   so no client-constructible string reaches the password step early.
// - Anti-enumeration: an unknown email and a wrong code fail identically, and
//   requestReset stays silent for unknown addresses.
// - A 6-digit code dies at the attempt cap regardless of later correctness.
//
// What this suite cannot cover: ResetOtpAttemptService's REQUIRES_NEW
// increment surviving the verify transaction's rollback — a mock can't
// observe a rollback (same caveat LoginAttemptService carries; see
// AuthLockoutIntegrationTest for the shape of the proof).
class PasswordResetServiceTest {

    private lateinit var userRepository: UserRepository
    private lateinit var tokenRepository: PasswordResetTokenRepository
    private lateinit var passwordEncoder: PasswordEncoder
    private lateinit var emailService: EmailService
    private lateinit var refreshTokenService: RefreshTokenService
    private lateinit var resetOtpAttemptService: ResetOtpAttemptService

    private val userId: UUID = UUID.randomUUID()
    private val email = "runner@test.local"

    @BeforeEach
    fun setUp() {
        userRepository = Mockito.mock(UserRepository::class.java)
        tokenRepository = Mockito.mock(PasswordResetTokenRepository::class.java)
        passwordEncoder = Mockito.mock(PasswordEncoder::class.java)
        emailService = Mockito.mock(EmailService::class.java)
        refreshTokenService = Mockito.mock(RefreshTokenService::class.java)
        resetOtpAttemptService = Mockito.mock(ResetOtpAttemptService::class.java)
    }

    // ─── requestReset ─────────────────────────────────────────────────────

    @Test
    fun `an unknown email is silent and sends nothing`() {
        Mockito.`when`(userRepository.findByEmail(email)).thenReturn(null)

        service().requestReset(email)

        Mockito.verify(emailService, Mockito.never()).sendPasswordResetEmail(anyArg(), anyArg())
        Mockito.verify(tokenRepository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `a request retires outstanding codes and mails a fresh 6-digit code`() {
        // Answers instead of ArgumentCaptor: captor.capture() returns null,
        // which trips Kotlin's null check when the argument type is non-null.
        stubUser()
        var savedRow: PasswordResetToken? = null
        Mockito.`when`(tokenRepository.save(anyArg<PasswordResetToken>())).thenAnswer { inv ->
            inv.getArgument<PasswordResetToken>(0).also { savedRow = it }
        }
        var mailedCode: String? = null
        Mockito.doAnswer { inv ->
            mailedCode = inv.getArgument(1)
            null
        }.`when`(emailService).sendPasswordResetEmail(anyArg(), anyArg())

        service().requestReset("  Runner@Test.Local  ")

        Mockito.verify(tokenRepository).invalidateOutstanding(eqArg(userId), anyArg())
        Mockito.verify(emailService).sendPasswordResetEmail(eqArg(email), anyArg())
        val code = assertNotNull(mailedCode, "no code was mailed")
        assertTrue(Regex("""\d{6}""").matches(code), "mailed code was '$code'")
        val saved = assertNotNull(savedRow, "no row was saved")
        assertEquals(OpaqueTokens.hash(code), saved.codeHash)
        assertNull(saved.tokenHash, "a fresh row must not be confirmable yet")
        assertEquals(0, saved.attempts)
    }

    // ─── verifyOtp ────────────────────────────────────────────────────────

    @Test
    fun `a correct code rotates the row into a continuation token and consumes the code`() {
        stubUser()
        val row = stubCodeRow()

        val continuation = service().verifyOtp(email, CODE)

        assertEquals(OpaqueTokens.hash(continuation), row.tokenHash)
        assertNull(row.codeHash, "the code is one-shot")
        Mockito.verify(tokenRepository).save(row)
    }

    @Test
    fun `an unknown email and a wrong code fail identically`() {
        // Anti-enumeration: the response must not reveal whether the account
        // exists. Same exception type, code, and message on both paths.
        stubUser()
        stubCodeRow()

        val wrongCode = assertThrows<ValidationException> { service().verifyOtp(email, "000000") }
        Mockito.`when`(userRepository.findByEmail("ghost@test.local")).thenReturn(null)
        val unknownEmail = assertThrows<ValidationException> { service().verifyOtp("ghost@test.local", CODE) }

        assertEquals(ErrorCodes.INVALID_RESET_CODE, wrongCode.code)
        assertEquals(wrongCode.code, unknownEmail.code)
        assertEquals(wrongCode.message, unknownEmail.message)
    }

    @Test
    fun `a wrong code records a failed attempt`() {
        stubUser()
        val row = stubCodeRow()

        assertThrows<ValidationException> { service().verifyOtp(email, "000000") }

        Mockito.verify(resetOtpAttemptService).recordFailure(row.id)
    }

    @Test
    fun `even the correct code is rejected at the attempt cap`() {
        stubUser()
        val row = stubCodeRow(attempts = PasswordResetService.MAX_OTP_ATTEMPTS)

        assertThrows<ValidationException> { service().verifyOtp(email, CODE) }

        assertNull(row.tokenHash, "a capped code must never rotate")
        assertNotNull(row.codeHash)
    }

    @Test
    fun `an expired code is rejected`() {
        stubUser()
        stubCodeRow(expiresAt = OffsetDateTime.now().minusMinutes(1))

        assertThrows<ValidationException> { service().verifyOtp(email, CODE) }
    }

    @Test
    fun `an already-consumed code is rejected`() {
        // codeHash is nulled on rotation; a replay of the same digits after a
        // successful verify must fail even though the row itself is live.
        stubUser()
        val row = stubCodeRow()
        row.codeHash = null
        row.tokenHash = OpaqueTokens.hash("some-continuation")

        assertThrows<ValidationException> { service().verifyOtp(email, CODE) }
    }

    // ─── confirmReset ─────────────────────────────────────────────────────

    @Test
    fun `the continuation token resets the password and revokes refresh tokens`() {
        val user = stubUser()
        val raw = stubContinuationRow()
        Mockito.`when`(passwordEncoder.encode("marathon-cebu-2027")).thenReturn("new-hash")

        service().confirmReset(raw, "marathon-cebu-2027")

        assertEquals("new-hash", user.passwordHash)
        Mockito.verify(refreshTokenService).revokeAllForUser(userId)
    }

    @Test
    fun `a reused continuation token is rejected`() {
        stubUser()
        val raw = stubContinuationRow(usedAt = OffsetDateTime.now().minusMinutes(1))

        val ex = assertThrows<ValidationException> { service().confirmReset(raw, "marathon-cebu-2027") }

        assertEquals(ErrorCodes.INVALID_RESET_TOKEN, ex.code)
    }

    @Test
    fun `an unknown token is rejected`() {
        Mockito.`when`(tokenRepository.findByTokenHash(anyArg())).thenReturn(null)

        val ex = assertThrows<ValidationException> { service().confirmReset("123456", "marathon-cebu-2027") }

        assertEquals(ErrorCodes.INVALID_RESET_TOKEN, ex.code)
    }

    @Test
    fun `the password screen still runs on confirm`() {
        val user = stubUser()
        val raw = stubContinuationRow()

        val ex = assertThrows<ValidationException> { service().confirmReset(raw, "password123") }

        assertEquals(ErrorCodes.WEAK_PASSWORD, ex.code)
        assertEquals("hashed", user.passwordHash)
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun service() = PasswordResetService(
        userRepository,
        tokenRepository,
        passwordEncoder,
        emailService,
        refreshTokenService,
        resetOtpAttemptService,
    )

    private fun stubUser(): User {
        val user = User(
            id = userId,
            email = email,
            passwordHash = "hashed",
            name = "Test Runner",
            role = Role.RUNNER,
        )
        Mockito.`when`(userRepository.findByEmail(email)).thenReturn(user)
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(user))
        return user
    }

    private fun stubCodeRow(
        attempts: Int = 0,
        expiresAt: OffsetDateTime = OffsetDateTime.now().plusMinutes(10),
    ): PasswordResetToken {
        val row = PasswordResetToken(
            userId = userId,
            codeHash = OpaqueTokens.hash(CODE),
            attempts = attempts,
            expiresAt = expiresAt,
        )
        Mockito.`when`(tokenRepository.findFirstByUserIdAndUsedAtIsNullOrderByCreatedAtDesc(userId))
            .thenReturn(row)
        return row
    }

    private fun stubContinuationRow(usedAt: OffsetDateTime? = null): String {
        val raw = "raw-continuation-token"
        val row = PasswordResetToken(
            userId = userId,
            tokenHash = OpaqueTokens.hash(raw),
            expiresAt = OffsetDateTime.now().plusMinutes(15),
            usedAt = usedAt,
        )
        Mockito.`when`(tokenRepository.findByTokenHash(OpaqueTokens.hash(raw))).thenReturn(row)
        return raw
    }

    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value

    private fun <T> anyArg(): T = Mockito.any()

    companion object {
        private const val CODE = "123456"
    }
}
