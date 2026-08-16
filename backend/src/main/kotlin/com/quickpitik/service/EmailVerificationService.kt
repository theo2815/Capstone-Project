package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.EmailVerificationToken
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EmailVerificationTokenRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.OpaqueTokens
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

/**
 * Proof that a registered address is reachable.
 *
 * **Advisory by design.** Redeeming a link stamps `users.email_verified_at` and
 * nothing else — no endpoint, upload, purchase, or role gate consults it. Both
 * clients sign a user in the instant `/auth/register` returns, so a hard gate
 * would drop them on a wall neither front end renders. Turning this into a
 * requirement is a cross-module decision, not a backend one.
 *
 * Mechanically this is [com.quickpitik.service.profile.EmailChangeService] with
 * the pending-address column removed: same opaque token hashed before storage,
 * same single-use redemption, same retire-the-previous-one behaviour on resend.
 * The difference is the expiry — 24 hours rather than one, because verifying is
 * not an errand the user is already mid-way through the way a password reset or
 * an inbox swap is.
 */
@Service
@Transactional
class EmailVerificationService(
    private val userRepository: UserRepository,
    private val tokenRepository: EmailVerificationTokenRepository,
    private val emailService: EmailService,
) {
    /**
     * Mint a link and mail it to the address on the account.
     *
     * Silent when the address is already verified: the only caller that can hit
     * that is a redelivery of the registration event, and re-mailing a
     * confirmed user would be noise. [resend] is the path that answers a human,
     * and it says so out loud.
     */
    fun sendVerification(userId: UUID) {
        val user = userRepository.findById(userId).orElse(null) ?: return
        if (user.emailVerifiedAt != null) return

        tokenRepository.invalidateOutstanding(userId, OffsetDateTime.now())

        val plaintext = OpaqueTokens.generate()
        tokenRepository.save(
            EmailVerificationToken(
                userId = user.id,
                tokenHash = OpaqueTokens.hash(plaintext),
                expiresAt = OffsetDateTime.now().plusHours(TOKEN_TTL_HOURS),
            ),
        )
        emailService.sendEmailVerification(user.email, plaintext)
    }

    /** Same as [sendVerification], but tells a signed-in caller when there is nothing to do. */
    fun resend(userId: UUID) {
        val user = userRepository.findById(userId).orElseThrow {
            ValidationException(message = "User not found", code = ErrorCodes.NOT_FOUND)
        }
        if (user.emailVerifiedAt != null) {
            throw ConflictException(
                code = ErrorCodes.EMAIL_ALREADY_VERIFIED,
                message = "That address is already verified.",
            )
        }
        sendVerification(userId)
    }

    /**
     * Redeem a link.
     *
     * One-shot, like every other opaque token here — a replay lands on the same
     * generic failure as an unknown one, so a used token can't be distinguished
     * from a fabricated one. Refresh tokens are deliberately NOT revoked:
     * verifying changes no credential and no sign-in identity, so signing every
     * device out would be a gratuitous punishment for doing the right thing.
     * (Contrast [com.quickpitik.service.profile.EmailChangeService.confirmChange],
     * where the sign-in address itself moves.)
     */
    fun confirm(rawToken: String) {
        val token = tokenRepository.findByTokenHash(OpaqueTokens.hash(rawToken))
            ?: throw ValidationException(
                message = "Invalid or expired verification link",
                code = ErrorCodes.INVALID_VERIFICATION_TOKEN,
            )
        if (!token.isUsable()) {
            throw ValidationException(
                message = "Verification link expired or already used",
                code = ErrorCodes.INVALID_VERIFICATION_TOKEN,
            )
        }
        val user = userRepository.findById(token.userId).orElseThrow {
            ValidationException(message = "User not found", code = ErrorCodes.NOT_FOUND)
        }

        user.emailVerifiedAt = OffsetDateTime.now()
        userRepository.save(user)

        token.usedAt = OffsetDateTime.now()
        tokenRepository.save(token)
    }

    private companion object {
        const val TOKEN_TTL_HOURS = 24L
    }
}
