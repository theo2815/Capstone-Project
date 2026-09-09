package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.JwtProperties
import com.quickpitik.config.PlatformProperties
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.JwtTokenProvider
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder
import java.time.Duration
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class StepUpServiceTest {

    private val userId = UUID.randomUUID()
    private val encoder = BCryptPasswordEncoder()
    private val tokenProvider = JwtTokenProvider(
        JwtProperties(
            secret = "step-up-test-secret-at-least-32-bytes-1234567890",
            accessTokenTtl = Duration.ofMinutes(15),
            refreshTokenTtl = Duration.ofDays(30),
        ),
    )
    private val userRepository = Mockito.mock(UserRepository::class.java)

    private fun service(stepUpEnabled: Boolean) = StepUpService(
        userRepository,
        encoder,
        tokenProvider,
        PlatformProperties(payoutStepUpEnabled = stepUpEnabled),
    )

    private fun givenPassword(raw: String) {
        val user = User(
            id = userId,
            email = "p@test.local",
            passwordHash = encoder.encode(raw),
            name = "P",
            role = Role.PHOTOGRAPHER,
        )
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(user))
    }

    @Test
    fun `reauth issues a step-up token for the correct password`() {
        givenPassword("correct-horse-battery-staple-42")
        val token = service(stepUpEnabled = false).reauth(userId, "correct-horse-battery-staple-42")
        assertEquals(userId, tokenProvider.parseStepUp(token))
    }

    @Test
    fun `reauth rejects the wrong password`() {
        givenPassword("correct-horse-battery-staple-42")
        val ex = assertFailsWith<UnauthorizedException> {
            service(stepUpEnabled = false).reauth(userId, "not-the-password")
        }
        assertEquals(ErrorCodes.INVALID_CREDENTIALS, ex.code)
    }

    @Test
    fun `enforce is a no-op when the flag is off`() {
        // No token, flag off → the shipped clients keep working.
        service(stepUpEnabled = false).enforce(userId, null)
    }

    @Test
    fun `enforce requires a valid token when the flag is on`() {
        val ex = assertFailsWith<ApiException> { service(stepUpEnabled = true).enforce(userId, null) }
        assertEquals(ErrorCodes.STEP_UP_REQUIRED, ex.code)
    }

    @Test
    fun `enforce passes with a valid step-up token when the flag is on`() {
        val token = tokenProvider.createStepUpToken(userId)
        service(stepUpEnabled = true).enforce(userId, token)
    }

    @Test
    fun `enforce rejects a step-up token minted for a different user`() {
        val someoneElse = tokenProvider.createStepUpToken(UUID.randomUUID())
        val ex = assertFailsWith<ApiException> { service(stepUpEnabled = true).enforce(userId, someoneElse) }
        assertEquals(ErrorCodes.STEP_UP_REQUIRED, ex.code)
    }
}
