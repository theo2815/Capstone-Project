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
    fun `password change consumes the shared limit before email change reaches its service`() {
        val profileService = Mockito.mock(ProfileService::class.java)
        val emailChangeService = Mockito.mock(EmailChangeService::class.java)
        val controller = MeProfileController(
            profileService,
            emailChangeService,
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
        Mockito.verifyNoInteractions(emailChangeService)
    }

    @Test
    fun `email change consumes the shared limit before password change reaches its service`() {
        val profileService = Mockito.mock(ProfileService::class.java)
        val emailChangeService = Mockito.mock(EmailChangeService::class.java)
        val controller = MeProfileController(
            profileService,
            emailChangeService,
            Bucket4jRateLimiter(
                RateLimitProperties(authLogin = RateLimitProperties.Policy(1, Duration.ofMinutes(15))),
                SimpleMeterRegistry(),
            ),
        )
        val principal = AuthPrincipal(UUID.randomUUID(), "runner@test.local", Role.RUNNER)

        controller.requestEmailChange(principal, EmailChangeRequest("next@test.local", "current-password"))

        val ex = assertFailsWith<ApiException> {
            controller.changePassword(principal, PasswordChangeRequest("current-password", "new-password-2026"))
        }
        assertEquals(ErrorCodes.RATE_LIMITED, ex.code)
        assertEquals(HttpStatus.TOO_MANY_REQUESTS, ex.status)
        Mockito.verifyNoInteractions(profileService)
    }
}
