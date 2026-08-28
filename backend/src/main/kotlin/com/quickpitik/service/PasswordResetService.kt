package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.PasswordResetToken
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PasswordResetTokenRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.OpaqueTokens
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime

// Three-step OTP flow (V37): requestReset mails a 6-digit code → verifyOtp
// trades the code for a one-shot continuation token → confirmReset consumes
// the token. A row is born with codeHash set and tokenHash NULL, and rotates
// on verification — so confirmReset's hash lookup can never match a row whose
// code was not verified, and the verify step cannot be bypassed.
@Service
@Transactional
class PasswordResetService(
    private val userRepository: UserRepository,
    private val passwordResetTokenRepository: PasswordResetTokenRepository,
    private val passwordEncoder: PasswordEncoder,
    private val emailService: EmailService,
    private val refreshTokenService: RefreshTokenService,
    private val resetOtpAttemptService: ResetOtpAttemptService,
) {
    companion object {
        const val MAX_OTP_ATTEMPTS = 5
        const val CODE_TTL_MINUTES = 10L
        const val CONTINUATION_TTL_MINUTES = 15L
    }

    fun requestReset(rawEmail: String) {
        val email = rawEmail.trim().lowercase()
        val user = userRepository.findByEmail(email) ?: return  // silent — anti-enumeration

        // Only the newest code is live: with a 10^6 code space, N outstanding
        // codes would multiply the guess surface.
        passwordResetTokenRepository.invalidateOutstanding(user.id, OffsetDateTime.now())

        val code = OpaqueTokens.generateOtp()
        val token = PasswordResetToken(
            userId = user.id,
            codeHash = OpaqueTokens.hash(code),
            expiresAt = OffsetDateTime.now().plusMinutes(CODE_TTL_MINUTES),
        )
        passwordResetTokenRepository.save(token)
        emailService.sendPasswordResetEmail(user.email, code)
    }

    fun verifyOtp(rawEmail: String, code: String): String {
        // Every failure below is IDENTICAL — unknown email must be
        // indistinguishable from a wrong code (anti-enumeration).
        fun fail(): Nothing =
            throw ValidationException("That code is invalid or has expired", ErrorCodes.INVALID_RESET_CODE)

        val user = userRepository.findByEmail(rawEmail.trim().lowercase()) ?: fail()
        val row = passwordResetTokenRepository
            .findFirstByUserIdAndUsedAtIsNullOrderByCreatedAtDesc(user.id) ?: fail()
        val codeHash = row.codeHash
        if (!row.isUsable() || codeHash == null || row.attempts >= MAX_OTP_ATTEMPTS) fail()
        if (codeHash != OpaqueTokens.hash(code)) {
            resetOtpAttemptService.recordFailure(row.id)  // REQUIRES_NEW — survives the throw
            fail()
        }

        val continuation = OpaqueTokens.generate()
        row.codeHash = null  // the code is one-shot
        row.tokenHash = OpaqueTokens.hash(continuation)
        row.expiresAt = OffsetDateTime.now().plusMinutes(CONTINUATION_TTL_MINUTES)
        passwordResetTokenRepository.save(row)
        return continuation
    }

    fun confirmReset(rawToken: String, newPassword: String) {
        val hash = OpaqueTokens.hash(rawToken)
        val token = passwordResetTokenRepository.findByTokenHash(hash)
            ?: throw ValidationException("Invalid or expired reset token", ErrorCodes.INVALID_RESET_TOKEN)
        if (!token.isUsable()) {
            throw ValidationException("Reset token expired or already used", ErrorCodes.INVALID_RESET_TOKEN)
        }
        val user = userRepository.findById(token.userId)
            .orElseThrow { ValidationException("User not found", ErrorCodes.USER_NOT_FOUND) }

        PasswordValidator.validate(newPassword, "newPassword")
        user.passwordHash = passwordEncoder.encode(newPassword)
        userRepository.save(user)

        token.usedAt = OffsetDateTime.now()
        passwordResetTokenRepository.save(token)

        refreshTokenService.revokeAllForUser(user.id)
    }
}
