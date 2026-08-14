package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.JwtProperties
import com.quickpitik.dto.auth.LoginRequest
import com.quickpitik.dto.auth.RefreshRequest
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.dto.auth.UserDto
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.profile.UserDtoMapper
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.security.authentication.BadCredentialsException
import org.springframework.security.crypto.password.PasswordEncoder
import java.time.Duration
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Three behaviours locked here, all of them holes the 2026-05-27 runner-flow
// audit found in the auth surface:
//
//  1. Suspension survives a refresh. F4 only blocked re-login, so a suspended
//     user could rotate refresh tokens indefinitely and was never locked out.
//  2. Login costs the same whether or not the email exists, so response time
//     can't be used to enumerate accounts.
//  3. Register actually consults PasswordValidator.
class AuthServiceTest {

    private lateinit var userRepository: UserRepository
    private lateinit var passwordEncoder: PasswordEncoder
    private lateinit var refreshTokenService: RefreshTokenService
    private lateinit var userDtoMapper: UserDtoMapper

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
        // The timing equalizer hashes a constant lazily; a null from the mock
        // would blow up on the non-null String property.
        Mockito.`when`(passwordEncoder.encode(anyArg())).thenReturn("\$2a\$12\$stub")
    }

    private fun service() = AuthService(
        userRepository,
        passwordEncoder,
        tokenProvider,
        refreshTokenService,
        userDtoMapper,
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

    private fun newUser(): User = User(
        id = UUID.randomUUID(),
        email = "runner@example.com",
        passwordHash = "\$2a\$12\$stub",
        name = "Test Runner",
        role = Role.RUNNER,
    )
}
