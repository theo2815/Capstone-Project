package com.quickpitik.service

import com.quickpitik.entity.PasswordResetToken
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PasswordResetTokenRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.OpaqueTokens
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime

@Service
@Transactional
class PasswordResetService(
    private val userRepository: UserRepository,
    private val passwordResetTokenRepository: PasswordResetTokenRepository,
    private val passwordEncoder: PasswordEncoder,
    private val emailService: EmailService,
    private val refreshTokenService: RefreshTokenService,
) {
    fun requestReset(rawEmail: String) {
        val email = rawEmail.trim().lowercase()
        val user = userRepository.findByEmail(email) ?: return  // silent — anti-enumeration

        val plaintext = OpaqueTokens.generate()
        val token = PasswordResetToken(
            userId = user.id,
            tokenHash = OpaqueTokens.hash(plaintext),
            expiresAt = OffsetDateTime.now().plusMinutes(15),
        )
        passwordResetTokenRepository.save(token)
        emailService.sendPasswordResetEmail(user.email, plaintext)
    }

    fun confirmReset(rawToken: String, newPassword: String) {
        val hash = OpaqueTokens.hash(rawToken)
        val token = passwordResetTokenRepository.findByTokenHash(hash)
            ?: throw ValidationException("Invalid or expired reset token", "INVALID_RESET_TOKEN")
        if (!token.isUsable()) {
            throw ValidationException("Reset token expired or already used", "INVALID_RESET_TOKEN")
        }
        val user = userRepository.findById(token.userId)
            .orElseThrow { ValidationException("User not found", "USER_NOT_FOUND") }

        PasswordValidator.validate(newPassword, "newPassword")
        user.passwordHash = passwordEncoder.encode(newPassword)
        userRepository.save(user)

        token.usedAt = OffsetDateTime.now()
        passwordResetTokenRepository.save(token)

        refreshTokenService.revokeAllForUser(user.id)
    }
}
