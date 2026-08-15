package com.quickpitik.service.profile

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.profile.EmailChangeRequest
import com.quickpitik.entity.EmailChangeToken
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EmailChangeTokenRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.OpaqueTokens
import com.quickpitik.service.EmailService
import com.quickpitik.service.RefreshTokenService
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

/**
 * Two-step change of a user's sign-in email.
 *
 * The order matters: the confirmation goes to the NEW address and `users.email`
 * only moves once that link is redeemed. A one-step swap would let anyone who
 * borrowed a signed-in session point the account at their own inbox, then use
 * "forgot password" to take it over outright — the old address would never hear
 * about it. Parking the address on a token instead means an unconfirmed request
 * changes nothing at all.
 *
 * All refresh tokens are revoked on confirmation. Sign-in identity has changed,
 * and the same reasoning as [com.quickpitik.service.PasswordResetService]
 * applies — any session that predates the change should re-authenticate.
 */
@Service
@Transactional
class EmailChangeService(
    private val userRepository: UserRepository,
    private val emailChangeTokenRepository: EmailChangeTokenRepository,
    private val passwordEncoder: PasswordEncoder,
    private val emailService: EmailService,
    private val refreshTokenService: RefreshTokenService,
) {
    fun requestChange(userId: UUID, request: EmailChangeRequest) {
        val user = userRepository.findById(userId).orElseThrow {
            UnauthorizedException(code = ErrorCodes.UNAUTHORIZED, message = "User not found")
        }
        if (!passwordEncoder.matches(request.currentPassword, user.passwordHash)) {
            throw UnauthorizedException(
                code = ErrorCodes.INVALID_CREDENTIALS,
                message = "Current password is incorrect",
            )
        }

        // Same normalisation every other entry point uses (AuthService:41,
        // PasswordResetService:23) so "Runner@Test.local " and "runner@test.local"
        // can't become two accounts.
        val newEmail = request.newEmail.trim().lowercase()
        if (newEmail == user.email) {
            throw ConflictException(
                code = ErrorCodes.SAME_EMAIL,
                message = "That is already your sign-in email.",
            )
        }
        if (userRepository.existsByEmail(newEmail)) {
            throw ConflictException(
                code = ErrorCodes.EMAIL_TAKEN,
                message = "That email is already registered to another account.",
            )
        }

        // Retire any earlier outstanding request so a mistyped address stops
        // being confirmable the moment a corrected one is issued.
        emailChangeTokenRepository.invalidateOutstanding(userId, OffsetDateTime.now())

        val plaintext = OpaqueTokens.generate()
        emailChangeTokenRepository.save(
            EmailChangeToken(
                userId = user.id,
                newEmail = newEmail,
                tokenHash = OpaqueTokens.hash(plaintext),
                // Longer than the 15-minute password-reset window on purpose:
                // this one requires the user to go and open a DIFFERENT inbox,
                // which is a slower errand than reading the mail they're already in.
                expiresAt = OffsetDateTime.now().plusHours(1),
            ),
        )
        emailService.sendEmailChangeConfirmation(newEmail, plaintext)
    }

    fun confirmChange(rawToken: String) {
        val token = emailChangeTokenRepository.findByTokenHash(OpaqueTokens.hash(rawToken))
            ?: throw ValidationException(
                message = "Invalid or expired confirmation link",
                code = ErrorCodes.INVALID_EMAIL_CHANGE_TOKEN,
            )
        if (!token.isUsable()) {
            throw ValidationException(
                message = "Confirmation link expired or already used",
                code = ErrorCodes.INVALID_EMAIL_CHANGE_TOKEN,
            )
        }
        val user = userRepository.findById(token.userId).orElseThrow {
            ValidationException(message = "User not found", code = ErrorCodes.NOT_FOUND)
        }

        // Re-check at redemption, not just at request time: the address was free
        // an hour ago, but someone may have registered it since. The DB UNIQUE on
        // users.email is the real backstop; this turns that 500 into a 409 the
        // user can act on.
        if (userRepository.existsByEmail(token.newEmail)) {
            throw ConflictException(
                code = ErrorCodes.EMAIL_TAKEN,
                message = "That email was registered to another account before you confirmed.",
            )
        }

        user.email = token.newEmail
        userRepository.save(user)

        token.usedAt = OffsetDateTime.now()
        emailChangeTokenRepository.save(token)

        refreshTokenService.revokeAllForUser(user.id)
    }
}
