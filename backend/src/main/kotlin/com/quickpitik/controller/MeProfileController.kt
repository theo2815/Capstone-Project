package com.quickpitik.controller

import com.quickpitik.dto.auth.MessageResponse
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.dto.profile.EmailChangeRequest
import com.quickpitik.dto.profile.PasswordChangeRequest
import com.quickpitik.dto.profile.ProfileUpdateRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.profile.EmailChangeService
import com.quickpitik.service.profile.ProfileService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import jakarta.validation.Valid
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.PutMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/v1/me")
class MeProfileController(
    private val profileService: ProfileService,
    private val emailChangeService: EmailChangeService,
    private val rateLimiter: RateLimiter,
) {
    @PutMapping("/profile")
    fun updateProfile(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: ProfileUpdateRequest,
    ): UserDto = profileService.updateName(principal.userId, body)

    @PutMapping("/password")
    fun changePassword(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: PasswordChangeRequest,
    ): MessageResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, principal.userId.toString())
        profileService.changePassword(principal.userId, body)
        return MessageResponse("Password updated. Sign in again on your other devices.")
    }

    // Step 1 of 2. Deliberately returns the same generic message whether or not
    // the address ends up confirmable, and does NOT change users.email — the
    // confirmation mail to the new address is what actually moves it. See
    // EmailChangeService for why the swap can't happen here.
    @PutMapping("/email")
    fun requestEmailChange(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: EmailChangeRequest,
    ): MessageResponse {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_AUTH_LOGIN, principal.userId.toString())
        emailChangeService.requestChange(principal.userId, principal.credentialVersion, body)
        return MessageResponse(
            "Check your new inbox — we sent a confirmation link. " +
                "Your sign-in email stays the same until you use it.",
        )
    }
}
