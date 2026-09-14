package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.RateLimitProperties
import com.quickpitik.dto.profile.EmailChangeRequest
import com.quickpitik.dto.profile.PasswordChangeRequest
import com.quickpitik.entity.Role
import com.quickpitik.exception.ApiException
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.profile.EmailChangeService
import com.quickpitik.service.profile.ProfileService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.time.Duration
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class MeProfileControllerTest {
    @Test
    fun `password change and email change share the authenticated user auth limit`() {
        val controller = MeProfileController(
            Mockito.mock(ProfileService::class.java),
            Mockito.mock(EmailChangeService::class.java),
            Bucket4jRateLimiter(
                RateLimitProperties(authLogin = RateLimitProperties.Policy(1, Duration.ofMinutes(15))),
                SimpleMeterRegistry(),
            ),
        )
        val principal = AuthPrincipal(UUID.randomUUID(), "runner@test.local", Role.RUNNER)

        controller.changePassword(principal, PasswordChangeRequest("current-password", "new-password-2026"))

        val ex = assertFailsWith<ApiException> {
            controller.requestEmailChange(principal, EmailChangeRequest("next@test.local", "current-password"))
        }
        assertEquals(ErrorCodes.RATE_LIMITED, ex.code)
        assertEquals(HttpStatus.TOO_MANY_REQUESTS, ex.status)
    }
}
