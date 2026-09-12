package com.quickpitik.service.profile

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.dto.profile.PasswordChangeRequest
import com.quickpitik.dto.profile.ProfileUpdateRequest
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.auth.PasswordValidator
import com.quickpitik.service.auth.RefreshTokenService
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

/**
 * Account self-edit (PUT /me/profile + PUT /me/password).
 *
 * Password change revokes all of the user's refresh tokens EXCEPT the one
 * the FE sends in `refreshToken` (the current device). Plan Phase E:
 * "Server revokes all OTHER refresh tokens (keep current session)." The FE
 * pulls the local refreshToken out of localStorage and includes it in the
 * PUT /me/password body so the BE can hash it and skip that one row when
 * revoking. If the body field is null/blank, we fall back to revoking
 * everything — that path keeps the endpoint usable for clients that
 * cannot supply the refresh token (e.g. an admin tool, a future device
 * recovery flow), at the cost of also kicking the current session.
 */
@Service
@Transactional
class ProfileService(
    private val userRepository: UserRepository,
    private val passwordEncoder: PasswordEncoder,
    private val refreshTokenService: RefreshTokenService,
    private val userDtoMapper: UserDtoMapper,
) {
    fun updateName(userId: UUID, request: ProfileUpdateRequest): UserDto {
        val name = request.name.trim()
        if (name.isEmpty()) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "name must not be blank",
                field = "name",
            )
        }
        if (name.length > 100) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "name must be at most 100 characters",
                field = "name",
            )
        }
        val user = userRepository.findById(userId).orElseThrow {
            UnauthorizedException(code = ErrorCodes.UNAUTHORIZED, message = "User not found")
        }
        user.name = name
        userRepository.save(user)
        return userDtoMapper.toDto(user)
    }

    fun changePassword(userId: UUID, request: PasswordChangeRequest) {
        val user = userRepository.findById(userId).orElseThrow {
            UnauthorizedException(code = ErrorCodes.UNAUTHORIZED, message = "User not found")
        }
        // Google sign-in proves identity at login, not permission to replace a
        // password from an existing session. Google-only users use OTP reset.
        if (request.currentPassword.isNullOrBlank()) {
            throw UnauthorizedException(
                code = ErrorCodes.INVALID_CREDENTIALS,
                message = "Enter your current password or verify your email through password reset",
            )
        }
        if (!passwordEncoder.matches(request.currentPassword, user.passwordHash)) {
            throw UnauthorizedException(
                code = ErrorCodes.INVALID_CREDENTIALS,
                message = "Current password is incorrect",
            )
        }
        if (passwordEncoder.matches(request.newPassword, user.passwordHash)) {
            throw ConflictException(
                code = ErrorCodes.SAME_PASSWORD,
                message = "New password must differ from current password",
            )
        }
        // Deliberately after the current-password check — an unauthenticated
        // caller learns nothing about our password rules.
        PasswordValidator.validate(request.newPassword, "newPassword")
        user.passwordHash = passwordEncoder.encode(request.newPassword)
        userRepository.save(user)
        refreshTokenService.revokeAllForUserExcept(userId, request.refreshToken)
    }
}
