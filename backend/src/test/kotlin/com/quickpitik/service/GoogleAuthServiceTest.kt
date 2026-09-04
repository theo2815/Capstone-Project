package com.quickpitik.service

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.GoogleAuthProperties
import com.quickpitik.config.JwtProperties
import com.quickpitik.dto.auth.GoogleLoginRequest
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
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.security.oauth2.jwt.BadJwtException
import org.springframework.security.oauth2.jwt.Jwt
import org.springframework.security.oauth2.jwt.JwtDecoder
import org.springframework.security.oauth2.jwt.JwtException
import java.time.Duration
import java.time.Instant
import java.time.OffsetDateTime
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// The Google door must be exactly as guarded as the password one. Locked here:
//
//  1. A suspended account cannot mint tokens via Google — neither through its
//     linked sub nor through the email auto-link (the F4 gate, third door).
//  2. The auto-link guard: a password account that never verified its email
//     gets its password rotated and every session revoked at link time, so a
//     pre-registered squatter keeps nothing. A verified account keeps its
//     password untouched.
//  3. A Google outage (JWKS unreachable) answers 503, never 401 — a 401 makes
//     both clients bounce a valid session to login.
//  4. New Google accounts skip the verification mail (the email is already
//     proven) and never carry a usable password.
//
// The decoder is mocked at the JwtDecoder interface — signature/iss/aud/exp
// live in GoogleJwtDecoderConfig's validators, which only a real key exchange
// can prove; the curl smoke test with a live GIS token covers that half.
class GoogleAuthServiceTest {

    private lateinit var userRepository: UserRepository
    private lateinit var passwordEncoder: PasswordEncoder
    private lateinit var refreshTokenService: RefreshTokenService
    private lateinit var userDtoMapper: UserDtoMapper
    private lateinit var loginAttemptService: LoginAttemptService
    private lateinit var eventPublisher: ApplicationEventPublisher
    private lateinit var googleJwtDecoder: JwtDecoder

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
        googleJwtDecoder = Mockito.mock(JwtDecoder::class.java)
        Mockito.`when`(passwordEncoder.encode(anyArg())).thenReturn("\$2a\$12\$rotated")
        Mockito.`when`(refreshTokenService.issue(anyArg())).thenReturn("refresh-plain")
        Mockito.`when`(userRepository.save(anyArg<User>())).thenAnswer { it.arguments[0] }
        Mockito.`when`(userDtoMapper.toDto(anyArg())).thenAnswer {
            val u = it.arguments[0] as User
            UserDto(id = u.id, email = u.email, name = u.name, role = u.role, createdAt = OffsetDateTime.now())
        }
    }

    private fun service(clientId: String = "test-client-id") = GoogleAuthService(
        userRepository,
        passwordEncoder,
        refreshTokenService,
        AuthService(
            userRepository,
            passwordEncoder,
            tokenProvider,
            refreshTokenService,
            userDtoMapper,
            loginAttemptService,
            eventPublisher,
        ),
        googleJwtDecoder,
        GoogleAuthProperties(clientId = clientId),
    )

    private fun googleJwt(
        sub: String = "google-sub-1",
        email: String = "juan.delacruz@gmail.com",
        emailVerified: Boolean? = true,
        name: String? = "Juan dela Cruz",
        picture: String? = null,
    ): Jwt {
        val builder = Jwt.withTokenValue("google-id-token")
            .header("alg", "RS256")
            .subject(sub)
            .issuedAt(Instant.now())
            .expiresAt(Instant.now().plusSeconds(3600))
            .claim("email", email)
        emailVerified?.let { builder.claim("email_verified", it) }
        name?.let { builder.claim("name", it) }
        picture?.let { builder.claim("picture", it) }
        return builder.build()
    }

    private fun newUser(
        email: String = "juan.delacruz@gmail.com",
        emailVerifiedAt: OffsetDateTime? = OffsetDateTime.now(),
    ) = User(
        email = email,
        passwordHash = "\$2a\$12\$existing",
        name = "Juan",
        role = Role.RUNNER,
        emailVerifiedAt = emailVerifiedAt,
    )

    @Test
    fun `a linked sub signs in without touching the row`() {
        val user = newUser()
        Mockito.`when`(googleJwtDecoder.decode("google-id-token")).thenReturn(googleJwt())
        Mockito.`when`(userRepository.findByGoogleSub("google-sub-1")).thenReturn(user)

        val result = service().login(GoogleLoginRequest("google-id-token"))

        assertEquals("refresh-plain", result.refreshToken)
        Mockito.verify(userRepository, Mockito.never()).save(anyArg<User>())
    }

    @Test
    fun `an unverified Google email is refused`() {
        Mockito.`when`(googleJwtDecoder.decode(anyArg())).thenReturn(googleJwt(emailVerified = false))

        val ex = assertFailsWith<UnauthorizedException> {
            service().login(GoogleLoginRequest("google-id-token"))
        }

        assertEquals(ErrorCodes.GOOGLE_EMAIL_UNVERIFIED, ex.code)
    }

    @Test
    fun `a bad token maps to 401 INVALID_GOOGLE_TOKEN`() {
        Mockito.`when`(googleJwtDecoder.decode(anyArg())).thenThrow(BadJwtException("expired"))

        val ex = assertFailsWith<UnauthorizedException> {
            service().login(GoogleLoginRequest("google-id-token"))
        }

        assertEquals(ErrorCodes.INVALID_GOOGLE_TOKEN, ex.code)
    }

    @Test
    fun `a JWKS outage maps to 503 not 401`() {
        Mockito.`when`(googleJwtDecoder.decode(anyArg())).thenThrow(JwtException("jwks fetch failed"))

        val ex = assertFailsWith<ApiException> {
            service().login(GoogleLoginRequest("google-id-token"))
        }

        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, ex.status)
        assertEquals(ErrorCodes.GOOGLE_AUTH_UNAVAILABLE, ex.code)
    }

    @Test
    fun `a blank client id answers 503 before touching Google`() {
        val ex = assertFailsWith<ApiException> {
            service(clientId = "").login(GoogleLoginRequest("google-id-token"))
        }

        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, ex.status)
        assertEquals(ErrorCodes.GOOGLE_AUTH_UNAVAILABLE, ex.code)
        Mockito.verify(googleJwtDecoder, Mockito.never()).decode(anyArg())
    }

    @Test
    fun `a brand-new Google user without a role gets ROLE_REQUIRED and no account`() {
        Mockito.`when`(googleJwtDecoder.decode(anyArg())).thenReturn(googleJwt())

        val ex = assertFailsWith<ApiException> {
            service().login(GoogleLoginRequest("google-id-token"))
        }

        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, ex.status)
        assertEquals(ErrorCodes.ROLE_REQUIRED, ex.code)
        Mockito.verify(userRepository, Mockito.never()).save(anyArg<User>())
    }

    @Test
    fun `a brand-new Google user cannot pick ADMIN`() {
        Mockito.`when`(googleJwtDecoder.decode(anyArg())).thenReturn(googleJwt())

        assertFailsWith<ValidationException> {
            service().login(GoogleLoginRequest("google-id-token", Role.ADMIN))
        }
        Mockito.verify(userRepository, Mockito.never()).save(anyArg<User>())
    }

    @Test
    fun `a brand-new Google user gets a verified account with no usable password and no mail`() {
        Mockito.`when`(googleJwtDecoder.decode(anyArg())).thenReturn(googleJwt())

        val result = service().login(GoogleLoginRequest("google-id-token", Role.RUNNER))

        val captor = ArgumentCaptor.forClass(User::class.java)
        Mockito.verify(userRepository).save(captor.capture())
        val saved = captor.value
        assertEquals("google-sub-1", saved.googleSub)
        assertEquals("juan.delacruz@gmail.com", saved.email)
        assertEquals("Juan dela Cruz", saved.name)
        assertEquals(Role.RUNNER, saved.role)
        assertNotNull(saved.emailVerifiedAt)
        // The hash comes from encoding 32 random bytes, never req material.
        assertEquals("\$2a\$12\$rotated", saved.passwordHash)
        // No verification mail — the address is already proven.
        Mockito.verify(eventPublisher, Mockito.never()).publishEvent(anyArg<Any>())
        assertEquals("refresh-plain", result.refreshToken)
    }

    @Test
    fun `a nameless token falls back to the email local-part`() {
        Mockito.`when`(googleJwtDecoder.decode(anyArg())).thenReturn(googleJwt(name = null))

        service().login(GoogleLoginRequest("google-id-token", Role.RUNNER))

        val captor = ArgumentCaptor.forClass(User::class.java)
        Mockito.verify(userRepository).save(captor.capture())
        assertEquals("juan.delacruz", captor.value.name)
    }

    @Test
    fun `linking to a verified password account leaves its password and sessions alone`() {
        val user = newUser()
        Mockito.`when`(googleJwtDecoder.decode(anyArg())).thenReturn(googleJwt())
        Mockito.`when`(userRepository.findByEmail("juan.delacruz@gmail.com")).thenReturn(user)

        service().login(GoogleLoginRequest("google-id-token"))

        assertEquals("google-sub-1", user.googleSub)
        assertEquals("\$2a\$12\$existing", user.passwordHash)
        Mockito.verify(refreshTokenService, Mockito.never()).revokeAllForUser(anyArg())
    }

    @Test
    fun `linking to an unverified password account rotates the password and revokes sessions`() {
        val user = newUser(emailVerifiedAt = null)
        Mockito.`when`(googleJwtDecoder.decode(anyArg())).thenReturn(googleJwt())
        Mockito.`when`(userRepository.findByEmail("juan.delacruz@gmail.com")).thenReturn(user)

        service().login(GoogleLoginRequest("google-id-token"))

        assertEquals("google-sub-1", user.googleSub)
        // The squatter's password is gone; the inbox owner OTP-resets a new one.
        assertEquals("\$2a\$12\$rotated", user.passwordHash)
        assertNotNull(user.emailVerifiedAt)
        Mockito.verify(refreshTokenService).revokeAllForUser(user.id)
    }

    @Test
    fun `a suspended account is refused through its linked sub`() {
        val user = newUser().apply { suspendedAt = OffsetDateTime.now() }
        Mockito.`when`(googleJwtDecoder.decode(anyArg())).thenReturn(googleJwt())
        Mockito.`when`(userRepository.findByGoogleSub("google-sub-1")).thenReturn(user)

        val ex = assertFailsWith<UnauthorizedException> {
            service().login(GoogleLoginRequest("google-id-token"))
        }

        assertEquals(ErrorCodes.ACCOUNT_SUSPENDED, ex.code)
    }

    @Test
    fun `a suspended account is refused through the email auto-link too`() {
        val user = newUser().apply { suspendedAt = OffsetDateTime.now() }
        Mockito.`when`(googleJwtDecoder.decode(anyArg())).thenReturn(googleJwt())
        Mockito.`when`(userRepository.findByEmail("juan.delacruz@gmail.com")).thenReturn(user)

        val ex = assertFailsWith<UnauthorizedException> {
            service().login(GoogleLoginRequest("google-id-token"))
        }

        assertEquals(ErrorCodes.ACCOUNT_SUSPENDED, ex.code)
        assertNull(user.googleSub)
    }

    @Test
    fun `the token email is normalized before matching`() {
        val user = newUser(email = "mixed@case.com")
        Mockito.`when`(googleJwtDecoder.decode(anyArg()))
            .thenReturn(googleJwt(email = "  MiXeD@Case.COM "))
        Mockito.`when`(userRepository.findByEmail("mixed@case.com")).thenReturn(user)

        service().login(GoogleLoginRequest("google-id-token"))

        assertEquals("google-sub-1", user.googleSub)
    }
}
