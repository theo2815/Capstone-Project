package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.JwtProperties
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.auth.RefreshRequest
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.profile.UserDtoMapper
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import org.springframework.security.authentication.BadCredentialsException
import org.springframework.security.crypto.password.PasswordEncoder
import java.time.Duration
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Four behaviours locked here. The first three are holes the 2026-05-27
// runner-flow audit found in the auth surface; the fourth arrived with V29:
//
//  1. Suspension survives a refresh. F4 only blocked re-login, so a suspended
//     user could rotate refresh tokens indefinitely and was never locked out.
//  2. Login costs the same whether or not the email exists, so response time
//     can't be used to enumerate accounts.
//  3. Register actually consults PasswordValidator.
//  4. A locked account is refused even when the password is right, and the
//     failure counter is driven from the login path.
//
// What these CANNOT cover: that recordFailure's write survives the login
// transaction's rollback. A mock records the call either way — only a real
// transaction shows the difference, which is AuthLockoutIntegrationTest's job.
class AuthServiceTest {

    private lateinit var userRepository: UserRepository
    private lateinit var passwordEncoder: PasswordEncoder
    private lateinit var refreshTokenService: RefreshTokenService
    private lateinit var userDtoMapper: UserDtoMapper
    private lateinit var loginAttemptService: LoginAttemptService
    private lateinit var eventPublisher: ApplicationEventPublisher

    private val tokenProvider = JwtTokenProvider(
        JwtProperties(
            secret = "test-secret-min-32-bytes-for-HS256-signing-purposes-only-do-not-use-in-prod",
            accessTokenTtl = Duration.ofMinutes(15),
            refreshTokenTtl = Duration.ofDays(7),
        ),
    )

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        userRepository = Mockito.mock(UserRepository::class.java)
        passwordEncoder = Mockito.mock(PasswordEncoder::class.java)
        refreshTokenService = Mockito.mock(RefreshTokenService::class.java)
        userDtoMapper = Mockito.mock(UserDtoMapper::class.java)
        loginAttemptService = Mockito.mock(LoginAttemptService::class.java)
        eventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)
        // The timing equalizer hashes a constant lazily; a null from the mock
        // would blow up on the non-null String property.
        Mockito.`when`(passwordEncoder.encode(anyArg())).thenReturn("\$2a\$12\$stub")
        // Mockito answers java.time.Duration with Duration.ZERO rather than
        // null, so an unstubbed lockRemaining() reads as "locked, 0 seconds
        // left" and every login in this class 429s. Default it to unlocked;
        // the lockout tests below override it.
        Mockito.`when`(loginAttemptService.lockRemaining(anyArg(), anyArg())).thenReturn(null)
    }

    private fun service() = AuthService(
        userRepository,
        passwordEncoder,
        tokenProvider,
        refreshTokenService,
        userDtoMapper,
        loginAttemptService,
        eventPublisher,
    )

    @Test
    fun `refresh is refused for a suspended user`() {
        val user = newUser().apply { suspendedAt = OffsetDateTime.now() }
        Mockito.`when`(refreshTokenService.validateAndRotate("stored-token"))
            .thenReturn(user.id to "rotated-token")
        Mockito.`when`(userRepository.findById(user.id)).thenReturn(Optional.of(user))

        val ex = assertFailsWith<UnauthorizedException> {
            service().refresh(RefreshRequest("stored-token"))
        }

        assertEquals(ErrorCodes.ACCOUNT_SUSPENDED, ex.code)
    }

    @Test
    fun `refresh still works for an active user`() {
        val user = newUser()
        Mockito.`when`(refreshTokenService.validateAndRotate("stored-token"))
            .thenReturn(user.id to "rotated-token")
        Mockito.`when`(userRepository.findById(user.id)).thenReturn(Optional.of(user))
        Mockito.`when`(userDtoMapper.toDto(anyArg())).thenReturn(
            UserDto(
                id = user.id,
                email = user.email,
                name = user.name,
                role = user.role,
                createdAt = OffsetDateTime.now(),
            ),
        )

        val response = service().refresh(RefreshRequest("stored-token"))

        assertEquals("rotated-token", response.refreshToken)
    }

    // An unknown email used to short-circuit before any hashing — ~10 ms versus
    // ~250 ms for a registered one, which is enough to enumerate accounts.
    @Test
    fun `login hashes against a dummy when the email is unknown`() {
        Mockito.`when`(userRepository.findByEmail("ghost@example.com")).thenReturn(null)

        assertFailsWith<BadCredentialsException> {
            service().login(LoginRequest(email = "ghost@example.com", password = "whatever123"))
        }

        Mockito.verify(passwordEncoder).matches(Mockito.eq("whatever123"), anyArg())
    }

    @Test
    fun `register refuses a password on the weak list`() {
        val ex = assertFailsWith<ValidationException> {
            service().register(
                RegisterRequest(
                    name = "Test Runner",
                    email = "runner@example.com",
                    password = "12345678",
                    role = Role.RUNNER,
                ),
            )
        }

        assertEquals(ErrorCodes.WEAK_PASSWORD, ex.code)
        Mockito.verify(userRepository, Mockito.never()).save(anyArg())
    }

    // ─── V29 lockout ──────────────────────────────────────────────────────

    // The point of the whole feature: guessing right on the attempt after the
    // lock must still get nothing, or the lock only inconveniences honest users.
    @Test
    fun `a locked account is refused even with the correct password`() {
        val user = newUser()
        Mockito.`when`(userRepository.findByEmail(user.email)).thenReturn(user)
        Mockito.`when`(passwordEncoder.matches(anyArg(), anyArg())).thenReturn(true)
        Mockito.`when`(loginAttemptService.lockRemaining(anyArg(), anyArg()))
            .thenReturn(Duration.ofMinutes(9))

        val ex = assertFailsWith<ApiException> {
            service().login(LoginRequest(email = user.email, password = "correct-horse-battery"))
        }

        assertEquals(ErrorCodes.ACCOUNT_LOCKED, ex.code)
        assertEquals(HttpStatus.TOO_MANY_REQUESTS, ex.status)
        // Retry-After lets the client show a countdown instead of guessing.
        assertEquals(541L, ex.retryAfterSeconds)
        Mockito.verify(refreshTokenService, Mockito.never()).issue(anyArg())
    }

    // The password is compared before the lock is consulted, so a locked
    // account can't answer faster than an unlocked one. Losing this re-opens
    // the timing channel the dummy-hash branch above exists to close.
    @Test
    fun `a locked account still pays for the password comparison`() {
        val user = newUser()
        Mockito.`when`(userRepository.findByEmail(user.email)).thenReturn(user)
        Mockito.`when`(loginAttemptService.lockRemaining(anyArg(), anyArg()))
            .thenReturn(Duration.ofMinutes(3))

        assertFailsWith<ApiException> {
            service().login(LoginRequest(email = user.email, password = "whatever123"))
        }

        Mockito.verify(passwordEncoder).matches(Mockito.eq("whatever123"), anyArg())
    }

    @Test
    fun `a wrong password counts toward the lock`() {
        val user = newUser()
        Mockito.`when`(userRepository.findByEmail(user.email)).thenReturn(user)
        Mockito.`when`(passwordEncoder.matches(anyArg(), anyArg())).thenReturn(false)

        assertFailsWith<BadCredentialsException> {
            service().login(LoginRequest(email = user.email, password = "wrong-one"))
        }

        Mockito.verify(loginAttemptService).recordFailure(user.id)
        Mockito.verify(loginAttemptService, Mockito.never()).recordSuccess(anyArg())
    }

    @Test
    fun `a successful login clears the streak`() {
        val user = newUser()
        Mockito.`when`(userRepository.findByEmail(user.email)).thenReturn(user)
        Mockito.`when`(passwordEncoder.matches(anyArg(), anyArg())).thenReturn(true)
        Mockito.`when`(refreshTokenService.issue(user.id)).thenReturn("issued-token")
        Mockito.`when`(userDtoMapper.toDto(anyArg())).thenReturn(
            UserDto(
                id = user.id,
                email = user.email,
                name = user.name,
                role = user.role,
                createdAt = OffsetDateTime.now(),
            ),
        )

        service().login(LoginRequest(email = user.email, password = "correct-horse-battery"))

        Mockito.verify(loginAttemptService).recordSuccess(user.id)
        Mockito.verify(loginAttemptService, Mockito.never()).recordFailure(anyArg())
    }

    // A suspended account must not have its streak cleared on the way out —
    // suspension is the harder gate and shouldn't hand back lockout headroom.
    @Test
    fun `a suspended user is refused before the streak is cleared`() {
        val user = newUser().apply { suspendedAt = OffsetDateTime.now() }
        Mockito.`when`(userRepository.findByEmail(user.email)).thenReturn(user)
        Mockito.`when`(passwordEncoder.matches(anyArg(), anyArg())).thenReturn(true)

        val ex = assertFailsWith<UnauthorizedException> {
            service().login(LoginRequest(email = user.email, password = "correct-horse-battery"))
        }

        assertEquals(ErrorCodes.ACCOUNT_SUSPENDED, ex.code)
        Mockito.verify(loginAttemptService, Mockito.never()).recordSuccess(anyArg())
    }

    private fun newUser(): User = User(
        id = UUID.randomUUID(),
        email = "runner@example.com",
        passwordHash = "\$2a\$12\$stub",
        name = "Test Runner",
        role = Role.RUNNER,
    )
}
