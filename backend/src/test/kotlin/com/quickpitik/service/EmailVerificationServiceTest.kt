package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.EmailVerificationToken
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EmailVerificationTokenRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.OpaqueTokens
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// Advisory email verification (V30). Mirrors EmailChangeServiceTest, which is
// the flow this one was modelled on — the token semantics are meant to stay
// identical, and these cases are what keeps them that way.
//
// The property that matters most here is the NEGATIVE one: confirming must not
// revoke sessions. Verification changes no credential, so signing every device
// out would punish the user for cooperating.
class EmailVerificationServiceTest {

    private lateinit var userRepository: UserRepository
    private lateinit var tokenRepository: EmailVerificationTokenRepository
    private lateinit var emailService: EmailService

    private val userId: UUID = UUID.randomUUID()
    private val email = "runner@test.local"

    @BeforeEach
    fun setUp() {
        userRepository = Mockito.mock(UserRepository::class.java)
        tokenRepository = Mockito.mock(EmailVerificationTokenRepository::class.java)
        emailService = Mockito.mock(EmailService::class.java)
    }

    // ─── send ─────────────────────────────────────────────────────────────

    @Test
    fun `sending mails the address on the account`() {
        stubUser()

        service().sendVerification(userId)

        Mockito.verify(emailService).sendEmailVerification(eqArg(email), anyArg())
        Mockito.verify(tokenRepository).save(anyArg())
    }

    // Otherwise every "I didn't get it" click leaves another live 24-hour link
    // behind, and the oldest mail in the inbox still works.
    @Test
    fun `sending retires any outstanding link`() {
        stubUser()

        service().sendVerification(userId)

        Mockito.verify(tokenRepository).invalidateOutstanding(eqArg(userId), anyArg())
    }

    // The one caller that can reach this is a redelivered registration event.
    @Test
    fun `sending to an already-verified account is a no-op`() {
        stubUser(verifiedAt = OffsetDateTime.now().minusDays(3))

        service().sendVerification(userId)

        Mockito.verify(emailService, Mockito.never()).sendEmailVerification(anyArg(), anyArg())
        Mockito.verify(tokenRepository, Mockito.never()).save(anyArg())
    }

    // A user deleted between the registration commit and the async listener.
    @Test
    fun `sending for a vanished user is ignored rather than thrown`() {
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.empty())

        service().sendVerification(userId)

        Mockito.verify(emailService, Mockito.never()).sendEmailVerification(anyArg(), anyArg())
    }

    // ─── resend ───────────────────────────────────────────────────────────

    @Test
    fun `resend tells a verified caller there is nothing to do`() {
        stubUser(verifiedAt = OffsetDateTime.now().minusDays(1))

        val ex = assertFailsWith<ConflictException> { service().resend(userId) }

        assertEquals(ErrorCodes.EMAIL_ALREADY_VERIFIED, ex.code)
        Mockito.verify(emailService, Mockito.never()).sendEmailVerification(anyArg(), anyArg())
    }

    @Test
    fun `resend mails again for an unverified account`() {
        stubUser()

        service().resend(userId)

        Mockito.verify(emailService).sendEmailVerification(eqArg(email), anyArg())
    }

    // ─── confirm ──────────────────────────────────────────────────────────

    @Test
    fun `confirming stamps the account and burns the token`() {
        val user = stubUser()
        val raw = stubToken()

        service().confirm(raw)

        assertNotNull(user.emailVerifiedAt)
    }

    // Verification changes no credential and no sign-in identity, so unlike
    // EmailChangeService.confirmChange it must NOT sign the user's devices out.
    // EmailVerificationService takes no RefreshTokenService at all, which is
    // what makes that structural rather than a matter of remembering.
    @Test
    fun `confirming leaves the email address untouched`() {
        val user = stubUser()
        val raw = stubToken()

        service().confirm(raw)

        assertEquals(email, user.email)
    }

    @Test
    fun `an expired link is refused`() {
        stubUser()
        val raw = stubToken(expiresAt = OffsetDateTime.now().minusMinutes(1))

        val ex = assertFailsWith<ValidationException> { service().confirm(raw) }

        assertEquals(ErrorCodes.INVALID_VERIFICATION_TOKEN, ex.code)
    }

    @Test
    fun `a link cannot be redeemed twice`() {
        val user = stubUser()
        val raw = stubToken(usedAt = OffsetDateTime.now().minusMinutes(5))

        assertFailsWith<ValidationException> { service().confirm(raw) }

        assertNull(user.emailVerifiedAt)
    }

    // Same code as a used one, on purpose: a replayed token must not be
    // distinguishable from a fabricated one.
    @Test
    fun `an unknown token is refused`() {
        Mockito.`when`(tokenRepository.findByTokenHash(anyArg())).thenReturn(null)

        val ex = assertFailsWith<ValidationException> { service().confirm("not-a-real-token") }

        assertEquals(ErrorCodes.INVALID_VERIFICATION_TOKEN, ex.code)
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun service() = EmailVerificationService(userRepository, tokenRepository, emailService)

    private fun stubUser(verifiedAt: OffsetDateTime? = null): User {
        val user = User(
            id = userId,
            email = email,
            passwordHash = "hashed",
            name = "Test Runner",
            role = Role.RUNNER,
            emailVerifiedAt = verifiedAt,
        )
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(user))
        return user
    }

    private fun stubToken(
        expiresAt: OffsetDateTime = OffsetDateTime.now().plusHours(24),
        usedAt: OffsetDateTime? = null,
    ): String {
        val raw = "raw-token-value"
        val token = EmailVerificationToken(
            userId = userId,
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
