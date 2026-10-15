package com.quickpitik.service.auth

import com.quickpitik.entity.PasswordResetToken
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PasswordResetTokenRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.email.EmailService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.security.crypto.password.PasswordEncoder
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

// Security audit 2026-10-14, auth/reset-otp-attempt-cap-resets-on-reissue.
// A stateful fake repository: a fully mocked one would prove nothing.
class PasswordResetAccountBudgetTest {
    private val user = User(email = "runner@test.local", passwordHash = "x", name = "Runner", role = Role.RUNNER)
    private val rows = mutableListOf<PasswordResetToken>()
    private val mailedCodes = mutableListOf<String>()
    private val users = Mockito.mock(UserRepository::class.java)
    private val tokens = Mockito.mock(PasswordResetTokenRepository::class.java)
    private val mail = Mockito.mock(EmailService::class.java)
    private val attempts = Mockito.mock(ResetOtpAttemptService::class.java)
    private lateinit var service: PasswordResetService

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        Mockito.`when`(users.findByEmail(user.email)).thenReturn(user)
        Mockito.`when`(tokens.save(anyArg())).thenAnswer { call ->
            (call.arguments[0] as PasswordResetToken).also { row -> if (rows.none { it.id == row.id }) rows += row }
        }
        Mockito.`when`(tokens.invalidateOutstanding(anyArg(), anyArg())).thenAnswer { call ->
            rows.filter { it.usedAt == null }.onEach { it.usedAt = call.getArgument<OffsetDateTime>(1) }.size
        }
        Mockito.`when`(tokens.findFirstByUserIdAndUsedAtIsNullOrderByCreatedAtDesc(user.id))
            .thenAnswer { rows.lastOrNull { it.usedAt == null } }
        Mockito.doAnswer { call -> rows.single { it.id == call.getArgument<UUID>(0) }.attempts += 1; null }
            .`when`(attempts).recordFailure(anyArg())
        Mockito.doAnswer { call -> mailedCodes += call.getArgument<String>(1); null }
            .`when`(mail).sendPasswordResetEmail(anyArg(), anyArg())
        service = PasswordResetService(users, tokens, Mockito.mock(PasswordEncoder::class.java), mail,
            Mockito.mock(RefreshTokenService::class.java), attempts)
    }

    private fun wrong(code: String) = ((code.toInt() + 1) % 1_000_000).toString().padStart(6, '0')

    @Disabled("SA-9 reproduction; SA-10 removes this line")
    @Test
    fun `wrong guesses across reissued codes run out a per-account budget`() {
        repeat(3) {
            service.requestReset(user.email)
            val code = mailedCodes.last()
            repeat(5) { assertFailsWith<ValidationException> { service.verifyOtp(user.email, wrong(code)) } }
        }

        service.requestReset(user.email)

        assertEquals(3, mailedCodes.size, "a fourth code was issued after 15 wrong guesses")
        assertEquals(0, user.failedLoginAttempts)
        assertNull(user.lockedUntil)
    }
}
