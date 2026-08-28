package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.GoogleAuthProperties
import com.quickpitik.dto.auth.AuthResponse
import com.quickpitik.dto.auth.GoogleLoginRequest
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.OpaqueTokens
import org.springframework.http.HttpStatus
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.security.oauth2.jwt.BadJwtException
import org.springframework.security.oauth2.jwt.Jwt
import org.springframework.security.oauth2.jwt.JwtDecoder
import org.springframework.security.oauth2.jwt.JwtException
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime

// "Continue with Google" for website + mobile: both exchange a Google-signed
// ID token at /auth/google for the normal QuickPitik pair. The decoder bean
// (GoogleJwtDecoderConfig) has already proven signature, timestamps, issuer,
// and audience by the time decode() returns — this service only decides what
// the verified identity maps to:
//
//   google_sub match          -> sign in.
//   email match (auto-link)   -> attach google_sub to that account. Google
//                                asserting email_verified is proof the caller
//                                owns the address, so linking is safe — EXCEPT
//                                that a password account which never verified
//                                its email may have been pre-registered by
//                                someone else entirely. For those, the old
//                                password is rotated to an unusable hash and
//                                every session revoked; the legitimate owner
//                                (who by definition controls the inbox) can
//                                mint a fresh password via the OTP reset flow.
//   no match                  -> new account. Google supplies no role, so the
//                                first attempt answers 422 ROLE_REQUIRED and
//                                the client re-POSTs with the picked role.
//                                No verification mail — the email is already
//                                proven, emailVerifiedAt is stamped directly.
@Service
@Transactional
class GoogleAuthService(
    private val userRepository: UserRepository,
    private val passwordEncoder: PasswordEncoder,
    private val refreshTokenService: RefreshTokenService,
    private val authService: AuthService,
    private val googleJwtDecoder: JwtDecoder,
    private val properties: GoogleAuthProperties,
) {
    fun login(req: GoogleLoginRequest): AuthResponse {
        if (properties.clientId.isBlank()) {
            throw ApiException(
                status = HttpStatus.SERVICE_UNAVAILABLE,
                code = ErrorCodes.GOOGLE_AUTH_UNAVAILABLE,
                message = "Google sign-in is not configured.",
            )
        }
        val jwt = decode(req.idToken)
        if (jwt.getClaimAsBoolean("email_verified") != true) {
            throw UnauthorizedException(
                "Google account email is not verified",
                ErrorCodes.GOOGLE_EMAIL_UNVERIFIED,
            )
        }
        val sub = jwt.subject
            ?: throw UnauthorizedException("Invalid Google token", ErrorCodes.INVALID_GOOGLE_TOKEN)
        // Same normalization as /auth/register — it is the join key for linking.
        val email = jwt.getClaimAsString("email")?.trim()?.lowercase()
            ?: throw UnauthorizedException("Invalid Google token", ErrorCodes.INVALID_GOOGLE_TOKEN)

        userRepository.findByGoogleSub(sub)?.let { user ->
            ensureNotSuspended(user)
            return authService.buildAuthResponse(user)
        }

        userRepository.findByEmail(email)?.let { user ->
            ensureNotSuspended(user)
            if (user.emailVerifiedAt == null) {
                // Pre-registration guard, per the class comment. OpaqueTokens
                // gives 32 random bytes — unguessable, never disclosed.
                user.passwordHash = passwordEncoder.encode(OpaqueTokens.generate())
                refreshTokenService.revokeAllForUser(user.id)
                user.emailVerifiedAt = OffsetDateTime.now()
            }
            user.googleSub = sub
            userRepository.save(user)
            return authService.buildAuthResponse(user)
        }

        val role = req.role ?: throw ApiException(
            status = HttpStatus.UNPROCESSABLE_ENTITY,
            code = ErrorCodes.ROLE_REQUIRED,
            message = "Choose a role to finish creating your account.",
            field = "role",
        )
        if (role == Role.ADMIN) {
            throw ValidationException("Cannot self-register as ADMIN", "INVALID_ROLE", "role")
        }
        val user = User(
            email = email,
            passwordHash = passwordEncoder.encode(OpaqueTokens.generate()),
            name = jwt.getClaimAsString("name")?.trim()?.takeIf { it.isNotBlank() }
                ?: email.substringBefore("@"),
            role = role,
            avatarUrl = jwt.getClaimAsString("picture"),
            emailVerifiedAt = OffsetDateTime.now(),
            googleSub = sub,
        )
        return authService.buildAuthResponse(userRepository.save(user))
    }

    // BadJwtException covers everything wrong with the TOKEN (signature,
    // expiry, issuer, audience, garbage input) -> 401. Any other JwtException
    // is OUR side failing to verify (JWKS fetch outage) -> 503, because a 401
    // makes both clients bounce the user to login over a Google blip.
    private fun decode(idToken: String): Jwt = try {
        googleJwtDecoder.decode(idToken)
    } catch (e: BadJwtException) {
        throw UnauthorizedException("Invalid Google token", ErrorCodes.INVALID_GOOGLE_TOKEN)
    } catch (e: JwtException) {
        throw ApiException(
            status = HttpStatus.SERVICE_UNAVAILABLE,
            code = ErrorCodes.GOOGLE_AUTH_UNAVAILABLE,
            message = "Google sign-in is temporarily unavailable. Try again shortly.",
        )
    }

    // Same gate login and refresh enforce (AuthService F4) — a suspended user
    // must not mint fresh tokens through the Google door either.
    private fun ensureNotSuspended(user: User) {
        if (user.suspendedAt != null) {
            throw UnauthorizedException("Account suspended", "ACCOUNT_SUSPENDED")
        }
    }
}
