package com.quickpitik.service

import com.quickpitik.dto.auth.AuthResponse
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.auth.RefreshRequest
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.dto.auth.toDto
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.security.JwtTokenProvider
import org.springframework.security.authentication.BadCredentialsException
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional

@Service
@Transactional
class AuthService(
    private val userRepository: UserRepository,
    private val passwordEncoder: PasswordEncoder,
    private val tokenProvider: JwtTokenProvider,
    private val refreshTokenService: RefreshTokenService,
) {
    fun register(req: RegisterRequest): AuthResponse {
        if (req.role == Role.ADMIN) {
            throw ValidationException("Cannot self-register as ADMIN", "INVALID_ROLE", "role")
        }
        val email = req.email.trim().lowercase()
        if (userRepository.existsByEmail(email)) {
            throw ConflictException("Email already registered", "EMAIL_TAKEN")
        }
        val user = User(
            email = email,
            passwordHash = passwordEncoder.encode(req.password),
            name = req.name.trim(),
            role = req.role,
        )
        val saved = userRepository.save(user)
        return buildAuthResponse(saved)
    }

    fun login(req: LoginRequest): AuthResponse {
        val email = req.email.trim().lowercase()
        val user = userRepository.findByEmail(email)
            ?: throw BadCredentialsException("Invalid email or password")
        if (!passwordEncoder.matches(req.password, user.passwordHash)) {
            throw BadCredentialsException("Invalid email or password")
        }
        return buildAuthResponse(user)
    }

    fun refresh(req: RefreshRequest): AuthResponse {
        val (userId, newRefreshToken) = refreshTokenService.validateAndRotate(req.refreshToken)
        val user = userRepository.findById(userId)
            .orElseThrow { UnauthorizedException("User not found", "USER_NOT_FOUND") }
        val accessToken = tokenProvider.createAccessToken(user)
        return AuthResponse(
            accessToken = accessToken,
            refreshToken = newRefreshToken,
            user = user.toDto(),
        )
    }

    @Transactional(readOnly = true)
    fun me(principal: AuthPrincipal): UserDto {
        val user = userRepository.findById(principal.userId)
            .orElseThrow { UnauthorizedException("User not found", "USER_NOT_FOUND") }
        return user.toDto()
    }

    private fun buildAuthResponse(user: User): AuthResponse {
        val accessToken = tokenProvider.createAccessToken(user)
        val refreshToken = refreshTokenService.issue(user.id)
        return AuthResponse(accessToken = accessToken, refreshToken = refreshToken, user = user.toDto())
    }
}
