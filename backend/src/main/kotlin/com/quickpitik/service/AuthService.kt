package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.auth.AuthResponse
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.auth.RefreshRequest
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.profile.UserDtoMapper
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
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
    private val userDtoMapper: UserDtoMapper,
    private val loginAttemptService: LoginAttemptService,
    private val eventPublisher: ApplicationEventPublisher,
) {
    // A BCrypt(12) hash of a throwaway constant, computed once on first use.
    // Matching against it costs the same as matching a real user's hash — see
    // login() for why that matters.
    private val dummyPasswordHash: String by lazy { passwordEncoder.encode(DUMMY_PASSWORD) }

    fun register(req: RegisterRequest): AuthResponse {
        if (req.role == Role.ADMIN) {
            throw ValidationException("Cannot self-register as ADMIN", "INVALID_ROLE", "role")
        }
        PasswordValidator.validate(req.password, "password")
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
        // AFTER_COMMIT + @Async, via EmailVerificationListener: a rolled-back
        // registration must not mail a link, and a slow Resend call must not
        // become sign-up latency. Verification is advisory — the tokens below
        // are issued regardless of whether that mail ever lands.
        eventPublisher.publishEvent(UserRegisteredEvent(saved.id))
        return buildAuthResponse(saved)
    }

    fun login(req: LoginRequest): AuthResponse {
        val email = req.email.trim().lowercase()
        val user = userRepository.findByEmail(email)
        if (user == null) {
            // Burn the same BCrypt work a real account costs before failing.
            // Returning immediately made the not-found branch ~10 ms against
            // ~250 ms for a registered email — a gap wide enough to enumerate
            // who has an account by timing the response alone.
            passwordEncoder.matches(req.password, dummyPasswordHash)
            throw BadCredentialsException("Invalid email or password")
        }
        // Computed before any branch below, never inside one. The lockout check
        // that follows must not be able to return earlier (or cheaper) than a
        // password check would have — that would re-open the timing channel the
        // dummy hash above exists to close.
        val passwordMatches = passwordEncoder.matches(req.password, user.passwordHash)

        // Lockout wins over a wrong password AND over a right one: an attacker
        // who guesses correctly on attempt six still gets nothing, which is the
        // entire point. Answering with a distinct code is safe here — /auth/register
        // already discloses whether an address is taken (EMAIL_TAKEN), so this
        // reveals nothing new, and silently refusing a correct password would
        // send real users to support instead of back in 15 minutes.
        loginAttemptService.lockRemaining(user)?.let { remaining ->
            val minutes = remaining.toMinutes() + 1
            throw ApiException(
                status = HttpStatus.TOO_MANY_REQUESTS,
                code = ErrorCodes.ACCOUNT_LOCKED,
                message = "Too many failed sign-in attempts. Try again in about $minutes minute(s).",
                retryAfterSeconds = remaining.toSeconds() + 1,
            )
        }

        if (!passwordMatches) {
            // Separate bean + REQUIRES_NEW: the throw below rolls this method's
            // transaction back, and an increment written inside it would go with
            // it. See LoginAttemptService.
            loginAttemptService.recordFailure(user.id)
            throw BadCredentialsException("Invalid email or password")
        }
        // F4 (2026-05-27): suspended users must not get fresh tokens.
        // Phase G admin suspension was previously a client-side-only block
        // — an existing token kept working until 15-min TTL expiry, and the
        // suspended user could log back in to mint a new one.
        if (user.suspendedAt != null) {
            throw UnauthorizedException("Account suspended", "ACCOUNT_SUSPENDED")
        }
        loginAttemptService.recordSuccess(user.id)
        return buildAuthResponse(user)
    }

    fun refresh(req: RefreshRequest): AuthResponse {
        val (userId, newRefreshToken) = refreshTokenService.validateAndRotate(req.refreshToken)
        val user = userRepository.findById(userId)
            .orElseThrow { UnauthorizedException("User not found", "USER_NOT_FOUND") }
        // The terminal half of the F4 suspension gate. F4 only blocked re-login,
        // so a suspended user could rotate refresh tokens indefinitely and was
        // never actually locked out. Load-bearing in two cases the admin-side
        // token revocation misses: a suspension written straight to the DB (how
        // this gets tested), and a refresh that commits alongside the revoke.
        // Throwing rolls back validateAndRotate above (this class is
        // @Transactional), so the caller's existing token isn't consumed.
        if (user.suspendedAt != null) {
            throw UnauthorizedException("Account suspended", "ACCOUNT_SUSPENDED")
        }
        val accessToken = tokenProvider.createAccessToken(user)
        return AuthResponse(
            accessToken = accessToken,
            refreshToken = newRefreshToken,
            user = userDtoMapper.toDto(user),
        )
    }

    @Transactional(readOnly = true)
    fun me(principal: AuthPrincipal): UserDto {
        val user = userRepository.findById(principal.userId)
            .orElseThrow { UnauthorizedException("User not found", "USER_NOT_FOUND") }
        return userDtoMapper.toDto(user)
    }

    fun logout(refreshToken: String?) {
        if (!refreshToken.isNullOrBlank()) {
            refreshTokenService.revoke(refreshToken)
        }
    }

    private fun buildAuthResponse(user: User): AuthResponse {
        val accessToken = tokenProvider.createAccessToken(user)
        val refreshToken = refreshTokenService.issue(user.id)
        return AuthResponse(
            accessToken = accessToken,
            refreshToken = refreshToken,
            user = userDtoMapper.toDto(user),
        )
    }

    private companion object {
        // Never a real credential — only ever fed to passwordEncoder.encode()
        // to produce a hash of the right cost for the timing-equalizer above.
        const val DUMMY_PASSWORD = "quickpitik-login-timing-equalizer"
    }
}
