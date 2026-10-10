package com.quickpitik.service.auth

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.auth.AuthResponse
import com.quickpitik.dto.auth.DesktopCodeResponse
import com.quickpitik.entity.DesktopAuthCode
import com.quickpitik.entity.Role
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.DesktopAuthCodeRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.OpaqueTokens
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.time.Clock
import java.time.Duration
import java.time.OffsetDateTime
import java.util.Base64
import java.util.UUID

/**
 * Connects the BatchMyPhotos desktop app to a photographer account (V85).
 *
 * The signed-in website mints a one-time code bound to a PKCE challenge; the
 * desktop redeems the code with the matching verifier and receives the normal
 * token pair through [AuthService.buildAuthResponse], the single place every
 * sign-in path mints tokens. Only the code's SHA-256 is stored, a code works
 * once (conditional update, see the repository), and every code failure
 * answers the same 401 so the caller learns nothing from the difference.
 */
@Service
@Transactional
class DesktopAuthService(
    private val desktopAuthCodeRepository: DesktopAuthCodeRepository,
    private val userRepository: UserRepository,
    private val authService: AuthService,
    private val clock: Clock,
) {
    fun mint(userId: UUID, codeChallenge: String): DesktopCodeResponse {
        if (!CHALLENGE.matches(codeChallenge)) {
            throw ValidationException(
                message = "codeChallenge must be a 43-character base64url string.",
                field = "codeChallenge",
            )
        }
        val user = userRepository.findById(userId).orElse(null)
            ?: throw UnauthorizedException("Unauthorized", ErrorCodes.UNAUTHORIZED)
        if (user.suspendedAt != null) {
            throw UnauthorizedException("Account suspended", ErrorCodes.ACCOUNT_SUSPENDED)
        }
        val now = OffsetDateTime.now(clock)
        val code = OpaqueTokens.generate()
        desktopAuthCodeRepository.save(
            DesktopAuthCode(
                userId = user.id,
                codeHash = OpaqueTokens.hash(code),
                codeChallenge = codeChallenge,
                expiresAt = now.plus(CODE_TTL),
                createdAt = now,
            ),
        )
        return DesktopCodeResponse(code = code, expiresInSeconds = CODE_TTL.seconds)
    }

    fun exchange(code: String, codeVerifier: String): AuthResponse {
        if (!CODE.matches(code) || !VERIFIER.matches(codeVerifier)) throw invalidCode()
        val now = OffsetDateTime.now(clock)
        val row = desktopAuthCodeRepository.findByCodeHash(OpaqueTokens.hash(code)) ?: throw invalidCode()
        if (!row.isUsable(now) || challengeOf(codeVerifier) != row.codeChallenge) throw invalidCode()
        if (desktopAuthCodeRepository.markUsed(row.id, now) != 1) throw invalidCode()
        val user = userRepository.findById(row.userId).orElse(null) ?: throw invalidCode()
        if (user.suspendedAt != null) {
            throw UnauthorizedException("Account suspended", ErrorCodes.ACCOUNT_SUSPENDED)
        }
        if (user.role != Role.PHOTOGRAPHER) throw invalidCode()
        return authService.buildAuthResponse(user)
    }

    private fun invalidCode() = UnauthorizedException(
        "This connect code is invalid or has expired.",
        ErrorCodes.INVALID_TOKEN,
    )

    companion object {
        val CODE_TTL: Duration = Duration.ofMinutes(2)
        private val CODE = Regex("^[A-Za-z0-9_-]{43}$")
        private val CHALLENGE = CODE
        private val VERIFIER = Regex("^[A-Za-z0-9_-]{43,128}$")

        // PKCE S256: base64url(SHA-256(verifier)) without padding.
        fun challengeOf(verifier: String): String {
            val digest = MessageDigest.getInstance("SHA-256")
                .digest(verifier.toByteArray(StandardCharsets.US_ASCII))
            return Base64.getUrlEncoder().withoutPadding().encodeToString(digest)
        }
    }
}
