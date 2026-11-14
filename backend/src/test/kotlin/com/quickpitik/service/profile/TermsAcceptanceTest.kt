package com.quickpitik.service.profile

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.JwtProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.admin.AdminMfaService
import com.quickpitik.service.auth.AuthService
import com.quickpitik.service.auth.LoginAttemptService
import com.quickpitik.service.auth.RefreshTokenService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.boot.context.properties.bind.Binder
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource
import org.springframework.boot.env.YamlPropertySourceLoader
import org.springframework.context.ApplicationEventPublisher
import org.springframework.core.env.StandardEnvironment
import org.springframework.core.io.ClassPathResource
import org.springframework.http.HttpStatus
import org.springframework.security.crypto.password.PasswordEncoder
import java.time.Clock
import java.time.Duration
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

// RV-L4 (plan 5.14, P5): who must accept the Terms again, and taking the acceptance.
class TermsAcceptanceTest {

    private val users = Mockito.mock(UserRepository::class.java)
    private val now = Instant.parse("2003-03-04T05:06:07Z")

    private fun legal(min: String? = "1.7", current: String = "1.7") =
        LegalProperties(currentVersion = current, photographerMinVersion = min)

    private fun mapper(legal: LegalProperties) = UserDtoMapper(Mockito.mock(StorageService::class.java), StorageProperties(), legal)

    private fun user(role: Role, accepted: String?) = User(email = "terms@test.local", passwordHash = "x", name = "Terms", role = role,
        termsVersion = accepted).also { Mockito.`when`(users.findById(it.id)).thenReturn(Optional.of(it)) }

    private fun profile(legal: LegalProperties) = ProfileService(users, Mockito.mock(PasswordEncoder::class.java),
        Mockito.mock(RefreshTokenService::class.java), mapper(legal), Mockito.mock(LoginAttemptService::class.java), legal,
        Clock.fixed(now, ZoneOffset.UTC))

    @Test
    fun `a photographer below the minimum must accept`() {
        val dto = mapper(legal()).toDto(user(Role.PHOTOGRAPHER, accepted = "1.5"))
        assertTrue(dto.termsAcceptanceRequired)
        assertEquals("1.5", dto.termsVersion)
        assertEquals("1.7", dto.legalCurrentVersion)
        assertTrue(mapper(legal()).toDto(user(Role.PHOTOGRAPHER, accepted = null)).termsAcceptanceRequired, "never accepted")
        assertFalse(mapper(legal()).toDto(user(Role.PHOTOGRAPHER, accepted = "1.7")).termsAcceptanceRequired)
    }

    @Test
    fun `a runner is never asked`() {
        assertFalse(mapper(legal()).toDto(user(Role.RUNNER, accepted = null)).termsAcceptanceRequired)
        assertFalse(mapper(legal()).toDto(user(Role.ADMIN, accepted = null)).termsAcceptanceRequired)
    }

    @Test
    fun `nobody is asked while the minimum is unset`() {
        assertFalse(mapper(legal(min = null)).toDto(user(Role.PHOTOGRAPHER, accepted = null)).termsAcceptanceRequired)
        assertFalse(mapper(legal(min = "")).toDto(user(Role.PHOTOGRAPHER, accepted = "1.0")).termsAcceptanceRequired)
    }

    @Test
    fun `a blank minimum in the yml counts as unset`() {
        val env = StandardEnvironment()
        YamlPropertySourceLoader().load("yml", ClassPathResource("application.yml")).forEach { env.propertySources.addLast(it) }
        val bound = Binder.get(env).bind("app.legal", LegalProperties::class.java).get()
        assertNull(bound.minVersionOrNull())
        assertTrue(javaClass.getResource("/application.yml")!!.readText().contains("photographer-min-version:"))

        val blank = Binder(MapConfigurationPropertySource(mapOf("app.legal.current-version" to "1.5", "app.legal.photographer-min-version" to " ")))
            .bind("app.legal", LegalProperties::class.java).get()
        assertNull(blank.minVersionOrNull())
        assertFalse(mapper(blank).toDto(user(Role.PHOTOGRAPHER, accepted = null)).termsAcceptanceRequired)
    }

    @Test
    fun `sign-in and me both carry the flag`() {
        val legal = legal()
        val refresh = Mockito.mock(RefreshTokenService::class.java)
        Mockito.`when`(refresh.issue(Mockito.any() ?: java.util.UUID.randomUUID())).thenReturn("refresh")
        val auth = AuthService(users, Mockito.mock(PasswordEncoder::class.java),
            JwtTokenProvider(JwtProperties(secret = "test-secret-min-32-bytes-for-HS256-signing-purposes-only-do-not-use-in-prod",
                accessTokenTtl = Duration.ofMinutes(15), refreshTokenTtl = Duration.ofDays(7))),
            refresh, mapper(legal), Mockito.mock(LoginAttemptService::class.java), Mockito.mock(ApplicationEventPublisher::class.java),
            Mockito.mock(AdminMfaService::class.java), legal)
        val photographer = user(Role.PHOTOGRAPHER, accepted = "1.5")

        assertTrue(auth.buildAuthResponse(photographer).user.termsAcceptanceRequired, "every sign-in path builds its response here")
        assertTrue(auth.me(AuthPrincipal(photographer.id, photographer.email, photographer.role)).termsAcceptanceRequired)
    }

    @Test
    fun `accepting the current version stamps it and clears the flag`() {
        val photographer = user(Role.PHOTOGRAPHER, accepted = "1.5")

        val dto = profile(legal()).acceptTerms(photographer.id, "1.7")

        assertEquals("1.7", photographer.termsVersion)
        assertEquals(OffsetDateTime.ofInstant(now, ZoneOffset.UTC), photographer.termsAcceptedAt)
        assertFalse(dto.termsAcceptanceRequired)
        assertEquals("1.7", dto.termsVersion)
    }

    @Test
    fun `accepting another version is refused`() {
        val photographer = user(Role.PHOTOGRAPHER, accepted = "1.5")

        val ex = assertFailsWith<ApiException> { profile(legal()).acceptTerms(photographer.id, "1.6") }

        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.CONSENT_VERSION_MISMATCH, ex.code)
        assertEquals("1.5", photographer.termsVersion)
        assertNull(photographer.termsAcceptedAt)
    }
}
