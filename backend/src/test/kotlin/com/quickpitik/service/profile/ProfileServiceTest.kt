package com.quickpitik.service.profile

import com.quickpitik.dto.profile.PasswordChangeRequest
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.RefreshTokenService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.mockito.Mockito
import org.springframework.security.crypto.password.PasswordEncoder
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals

class ProfileServiceTest {

    private lateinit var userRepository: UserRepository
    private lateinit var passwordEncoder: PasswordEncoder
    private lateinit var refreshTokenService: RefreshTokenService
    private lateinit var userDtoMapper: UserDtoMapper

    private val userId: UUID = UUID.randomUUID()

    @BeforeEach
    fun setUp() {
        userRepository = Mockito.mock(UserRepository::class.java)
        passwordEncoder = Mockito.mock(PasswordEncoder::class.java)
        refreshTokenService = Mockito.mock(RefreshTokenService::class.java)
        userDtoMapper = Mockito.mock(UserDtoMapper::class.java)
    }

    private fun service() = ProfileService(
        userRepository,
        passwordEncoder,
        refreshTokenService,
        userDtoMapper,
    )

    private fun stubUser(googleSub: String? = null, passwordHash: String = "old-hash"): User {
        val user = User(
            id = userId,
            email = "user@test.local",
            passwordHash = passwordHash,
            name = "Test User",
            role = Role.RUNNER,
            googleSub = googleSub,
        )
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(user))
        return user
    }

    @Test
    fun `standard user without current password is rejected`() {
        stubUser(googleSub = null)

        assertThrows<UnauthorizedException> {
            service().changePassword(userId, PasswordChangeRequest(currentPassword = null, newPassword = "NewPassword123!"))
        }
    }

    @Test
    fun `standard user with wrong current password is rejected`() {
        stubUser(googleSub = null)
        Mockito.`when`(passwordEncoder.matches("wrong-password", "old-hash")).thenReturn(false)

        assertThrows<UnauthorizedException> {
            service().changePassword(userId, PasswordChangeRequest(currentPassword = "wrong-password", newPassword = "NewPassword123!"))
        }
    }

    @Test
    fun `standard user with correct current password successfully changes password`() {
        val user = stubUser(googleSub = null)
        Mockito.`when`(passwordEncoder.matches("correct-password", "old-hash")).thenReturn(true)
        Mockito.`when`(passwordEncoder.matches("NewPassword123!", "old-hash")).thenReturn(false)
        Mockito.`when`(passwordEncoder.encode("NewPassword123!")).thenReturn("new-hash")

        service().changePassword(
            userId,
            PasswordChangeRequest(currentPassword = "correct-password", newPassword = "NewPassword123!", refreshToken = "token-123"),
        )

        assertEquals("new-hash", user.passwordHash)
        Mockito.verify(userRepository).save(user)
        Mockito.verify(refreshTokenService).revokeAllForUserExcept(userId, "token-123")
    }

    @Test
    fun `Google OAuth user can set initial password without current password`() {
        val user = stubUser(googleSub = "google-sub-xyz", passwordHash = "opaque-random-hash")
        Mockito.`when`(passwordEncoder.encode("NewPassword123!")).thenReturn("new-hash")

        service().changePassword(
            userId,
            PasswordChangeRequest(currentPassword = null, newPassword = "NewPassword123!", refreshToken = "token-456"),
        )

        assertEquals("new-hash", user.passwordHash)
        Mockito.verify(userRepository).save(user)
        Mockito.verify(refreshTokenService).revokeAllForUserExcept(userId, "token-456")
    }

    @Test
    fun `password same as existing is rejected`() {
        stubUser(googleSub = null)
        Mockito.`when`(passwordEncoder.matches("correct-password", "old-hash")).thenReturn(true)
        Mockito.`when`(passwordEncoder.matches("SamePassword123!", "old-hash")).thenReturn(true)

        assertThrows<ConflictException> {
            service().changePassword(
                userId,
                PasswordChangeRequest(currentPassword = "correct-password", newPassword = "SamePassword123!"),
            )
        }
    }
}
