package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.JwtProperties
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.RateLimitProperties
import com.quickpitik.dto.admin.MfaCodeRequest
import com.quickpitik.entity.Role
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.StepUpService
import com.quickpitik.service.admin.AdminMfaService
import com.quickpitik.service.admin.AdminPayoutService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import org.springframework.security.crypto.password.PasswordEncoder
import java.time.Duration
import java.util.UUID
import kotlin.test.*

class PayoutSecurityControllerTest {
    private val principal = AuthPrincipal(UUID.randomUUID(), "admin@test.local", Role.ADMIN)

    @Test fun `transfer initiation requires a valid proof for the acting admin`() {
        val payouts = Mockito.mock(AdminPayoutService::class.java)
        val jwt = JwtTokenProvider(JwtProperties(secret = "test-only-step-up-key-at-least-32-bytes-long",
            accessTokenTtl = Duration.ofMinutes(15), refreshTokenTtl = Duration.ofDays(7)))
        val stepUp = StepUpService(Mockito.mock(UserRepository::class.java), Mockito.mock(PasswordEncoder::class.java),
            jwt, PlatformProperties(payoutStepUpEnabled = true))
        val controller = AdminPayoutsController(payouts, stepUp)
        for (proof in listOf(null, jwt.createStepUpToken(UUID.randomUUID()))) {
            assertEquals(ErrorCodes.STEP_UP_REQUIRED,
                assertFailsWith<ApiException> { controller.markProcessing(principal, "PAY-TEST", proof) }.code)
        }
        Mockito.verifyNoInteractions(payouts)
        controller.markProcessing(principal, "PAY-TEST", jwt.createStepUpToken(principal.userId))
        Mockito.verify(payouts).markProcessing(principal.userId, "PAY-TEST")
    }

    @Test fun `confirm and disable share a per-admin guessing limit`() {
        val mfa = Mockito.mock(AdminMfaService::class.java)
        Mockito.`when`(mfa.confirm(principal.userId, "invalid"))
            .thenThrow(ApiException(HttpStatus.UNAUTHORIZED, ErrorCodes.INVALID_MFA_CODE, "Invalid code"))
        val limiter = Bucket4jRateLimiter(RateLimitProperties(
            authLogin = RateLimitProperties.Policy(1, Duration.ofMinutes(15)),
        ), SimpleMeterRegistry())
        val controller = AdminMfaController(mfa, limiter)
        assertEquals(HttpStatus.UNAUTHORIZED,
            assertFailsWith<ApiException> { controller.confirm(principal, MfaCodeRequest("invalid")) }.status)
        assertEquals(HttpStatus.TOO_MANY_REQUESTS,
            assertFailsWith<ApiException> { controller.disable(principal, MfaCodeRequest("invalid")) }.status)
        Mockito.verify(mfa, Mockito.never()).disable(principal.userId, "invalid")
    }
}
