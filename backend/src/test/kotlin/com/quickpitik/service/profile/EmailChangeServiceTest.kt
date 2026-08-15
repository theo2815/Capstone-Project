package com.quickpitik.service.profile

import com.quickpitik.dto.profile.EmailChangeRequest
import com.quickpitik.entity.EmailChangeToken
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EmailChangeTokenRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.OpaqueTokens
import com.quickpitik.service.EmailService
import com.quickpitik.service.RefreshTokenService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.mockito.Mockito
import org.springframework.security.crypto.password.PasswordEncoder
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals

// Backend half of the change-email gap the website agent filed (their side
// closed 2026-08-14 by renaming the slab to "Sign-in email"; there was no
// PUT /me/email to call).
//
// The security property under test is verify-then-swap: a request must not
// move users.email. Whoever controls the sign-in address controls password
// reset, so a one-step swap would let a borrowed session walk the account away
// and the real owner would never be told.
class EmailChangeServiceTest {

    private lateinit var userRepository: UserRepository
    private lateinit var tokenRepository: EmailChangeTokenRepository
    private lateinit var passwordEncoder: PasswordEncoder
    private lateinit var emailService: EmailService
    private lateinit var refreshTokenService: RefreshTokenService

    private val userId: UUID = UUID.randomUUID()
    private val currentEmail = "runner@test.local"
    private val newEmail = "new-address@test.local"

    @BeforeEach
    fun setUp() {
        userRepository = Mockito.mock(UserRepository::class.java)
        tokenRepository = Mockito.mock(EmailChangeTokenRepository::class.java)
        passwordEncoder = Mockito.mock(PasswordEncoder::class.java)
        emailService = Mockito.mock(EmailService::class.java)
        refreshTokenService = Mockito.mock(RefreshTokenService::class.java)
    }

    // ─── request ──────────────────────────────────────────────────────────

    @Test
    fun `requesting a change does not touch the current sign-in email`() {
        // The whole point of the two-step flow.
        val user = stubUser()
        stubPasswordMatches(true)

        service().requestChange(userId, request())

        assertEquals(currentEmail, user.email)
        Mockito.verify(userRepository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `the confirmation goes to the new address, never the old one`() {
        // Receiving it is the proof of control. Sending to the old address
        // would prove nothing about the requester.
        stubUser()
        stubPasswordMatches(true)

        service().requestChange(userId, request())

        Mockito.verify(emailService).sendEmailChangeConfirmation(eqArg(newEmail), anyArg())
    }

    @Test
    fun `a wrong current password is rejected before anything is sent`() {
        stubUser()
        stubPasswordMatches(false)

        assertThrows<UnauthorizedException> { service().requestChange(userId, request()) }

        Mockito.verify(emailService, Mockito.never()).sendEmailChangeConfirmation(anyArg(), anyArg())
        Mockito.verify(tokenRepository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `an address already registered to someone else is refused`() {
        stubUser()
        stubPasswordMatches(true)
        Mockito.`when`(userRepository.existsByEmail(newEmail)).thenReturn(true)

        assertThrows<ConflictException> { service().requestChange(userId, request()) }

        Mockito.verify(emailService, Mockito.never()).sendEmailChangeConfirmation(anyArg(), anyArg())
    }

    @Test
    fun `asking for the address you already have is refused`() {
        stubUser()
        stubPasswordMatches(true)

        assertThrows<ConflictException> {
            service().requestChange(userId, request(newEmail = currentEmail))
        }
    }

    @Test
    fun `the address is normalised so casing and spacing cannot fork an account`() {
        stubUser()
        stubPasswordMatches(true)

        service().requestChange(userId, request(newEmail = "  NEW-Address@Test.Local  "))

        Mockito.verify(emailService).sendEmailChangeConfirmation(eqArg(newEmail), anyArg())
    }

    @Test
    fun `a mistyped earlier request stops being confirmable`() {
        // Otherwise a typo'd address keeps a live link for an hour.
        stubUser()
        stubPasswordMatches(true)

        service().requestChange(userId, request())

        Mockito.verify(tokenRepository).invalidateOutstanding(eqArg(userId), anyArg())
    }

    // ─── confirm ──────────────────────────────────────────────────────────

    @Test
    fun `confirming swaps the email and forces a re-login`() {
        val user = stubUser()
        val raw = stubUsableToken()

        service().confirmChange(raw)

        assertEquals(newEmail, user.email)
        Mockito.verify(refreshTokenService).revokeAllForUser(userId)
    }

    @Test
    fun `an expired link is refused`() {
        stubUser()
        val raw = stubToken(expiresAt = OffsetDateTime.now().minusMinutes(1))

        assertThrows<ValidationException> { service().confirmChange(raw) }
    }

    @Test
    fun `a link cannot be redeemed twice`() {
        stubUser()
        val raw = stubToken(usedAt = OffsetDateTime.now().minusMinutes(5))

        assertThrows<ValidationException> { service().confirmChange(raw) }
    }

    @Test
    fun `an unknown token is refused`() {
        Mockito.`when`(tokenRepository.findByTokenHash(anyArg())).thenReturn(null)

        assertThrows<ValidationException> { service().confirmChange("not-a-real-token") }
    }

    @Test
    fun `an address claimed during the wait yields a conflict, not a constraint crash`() {
        // An hour passes between request and redemption; someone else can
        // register the address in between. The users.email UNIQUE is the real
        // backstop — this turns its 500 into an actionable 409.
        stubUser()
        val raw = stubUsableToken()
        Mockito.`when`(userRepository.existsByEmail(newEmail)).thenReturn(true)

        assertThrows<ConflictException> { service().confirmChange(raw) }
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun service() = EmailChangeService(
        userRepository,
        tokenRepository,
        passwordEncoder,
        emailService,
        refreshTokenService,
    )

    private fun request(newEmail: String = this.newEmail) =
        EmailChangeRequest(newEmail = newEmail, currentPassword = "correct-horse")

    private fun stubUser(): User {
        val user = User(
            id = userId,
            email = currentEmail,
            passwordHash = "hashed",
            name = "Test Runner",
            role = Role.RUNNER,
        )
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(user))
        return user
    }

    private fun stubPasswordMatches(matches: Boolean) {
        Mockito.`when`(passwordEncoder.matches(anyArg(), anyArg())).thenReturn(matches)
    }

    private fun stubUsableToken(): String = stubToken()

    private fun stubToken(
        expiresAt: OffsetDateTime = OffsetDateTime.now().plusHours(1),
        usedAt: OffsetDateTime? = null,
    ): String {
        val raw = "raw-token-value"
        val token = EmailChangeToken(
            userId = userId,
            newEmail = newEmail,
            tokenHash = OpaqueTokens.hash(raw),
            expiresAt = expiresAt,
            usedAt = usedAt,
        )
        Mockito.`when`(tokenRepository.findByTokenHash(OpaqueTokens.hash(raw))).thenReturn(token)
        return raw
    }

    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value

    private fun <T> anyArg(): T = Mockito.any()
}
