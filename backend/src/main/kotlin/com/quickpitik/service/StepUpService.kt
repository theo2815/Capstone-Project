package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PlatformProperties
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.JwtTokenProvider
import org.springframework.http.HttpStatus
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Service
import java.util.UUID

/**
 * Step-up re-authentication for sensitive money actions (payout-account change,
 * admin mark-paid). The user proves possession of their password again and gets
 * a short-lived token; the sensitive endpoint then requires that token in an
 * X-Step-Up header. Gated by PAYOUT_STEP_UP_ENABLED so it's a no-op until both
 * clients send the header (shipped clients don't 403).
 */
@Service
class StepUpService(
    private val userRepository: UserRepository,
    private val passwordEncoder: PasswordEncoder,
    private val tokenProvider: JwtTokenProvider,
    private val platformProperties: PlatformProperties,
) {
    /**
     * Verify the authenticated user's password and mint a step-up token. Wrong
     * password (or an account with no usable password, e.g. Google-only) → 401
     * with the same generic code as login, and no token.
     */
    fun reauth(userId: UUID, password: String): String {
        val user = userRepository.findById(userId).orElseThrow {
            UnauthorizedException(code = ErrorCodes.INVALID_CREDENTIALS, message = "Invalid credentials")
        }
        if (!passwordEncoder.matches(password, user.passwordHash)) {
            throw UnauthorizedException(code = ErrorCodes.INVALID_CREDENTIALS, message = "Invalid credentials")
        }
        return tokenProvider.createStepUpToken(userId)
    }

    /**
     * Enforce step-up on a sensitive action. No-op when the feature is off;
     * otherwise the X-Step-Up token must be valid, unexpired, and belong to the
     * acting user.
     */
    fun enforce(userId: UUID, stepUpToken: String?) {
        if (!platformProperties.payoutStepUpEnabled) return
        val subject = stepUpToken?.let { tokenProvider.parseStepUp(it) }
        if (subject == null || subject != userId) {
            throw ApiException(
                status = HttpStatus.FORBIDDEN,
                code = ErrorCodes.STEP_UP_REQUIRED,
                message = "Re-enter your password to continue.",
            )
        }
    }
}
