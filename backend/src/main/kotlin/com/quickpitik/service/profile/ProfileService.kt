package com.quickpitik.service.profile

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.dto.profile.PasswordChangeRequest
import com.quickpitik.dto.profile.ProfileUpdateRequest
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.RefreshTokenService
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

/**
 * Account self-edit (PUT /me/profile + PUT /me/password).
 *
 * Password change revokes all of the user's refresh tokens — this kicks
 * sessions on every device including the current one. The current access
 * token (15-min TTL) keeps working until it expires; after that the FE
 * detects the 401, prompts re-login, and the user gets a fresh refresh
 * token. The plan's "keep current session" intent is best served this way
 * given the FE doesn't send the refresh token in the password-change body.
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
        user.passwordHash = passwordEncoder.encode(request.newPassword)
        userRepository.save(user)
        refreshTokenService.revokeAllForUser(userId)
    }
}
