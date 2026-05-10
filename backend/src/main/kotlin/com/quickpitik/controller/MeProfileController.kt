package com.quickpitik.controller

import com.quickpitik.dto.auth.UserDto
import com.quickpitik.dto.profile.PasswordChangeRequest
import com.quickpitik.dto.profile.ProfileUpdateRequest
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.profile.ProfileService
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
    ): Map<String, String> {
        profileService.changePassword(principal.userId, body)
        return mapOf("message" to "Password updated. Sign in again on your other devices.")
    }
}
