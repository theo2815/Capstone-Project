package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

// The one gate every photographer dashboard action (create coupon/event,
// upload) routes through. Pins the two 403 codes the clients branch on and the
// fail-closed default: no settings row reads as "not verified".
class PhotographerAccessGuardTest {

    private val photographerId = UUID.randomUUID()
    private lateinit var userRepository: UserRepository
    private lateinit var settingsRepository: PhotographerSettingsRepository
    private lateinit var guard: PhotographerAccessGuard

    @BeforeEach
    fun setUp() {
        userRepository = Mockito.mock(UserRepository::class.java)
        settingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        guard = PhotographerAccessGuard(userRepository, settingsRepository)
    }

    private fun stubUser(suspended: Boolean = false) {
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(
            Optional.of(
                User(
                    id = photographerId,
                    email = "p@test.local",
                    passwordHash = "x",
                    name = "Aira",
                    role = Role.PHOTOGRAPHER,
                    suspendedAt = if (suspended) OffsetDateTime.now() else null,
                ),
            ),
        )
    }

    private fun stubSettings(status: VerificationStatus?) {
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(
            Optional.ofNullable(
                status?.let { PhotographerSettings(userId = photographerId, verificationStatus = it) },
            ),
        )
    }

    @Test
    fun `an approved, un-suspended photographer passes and gets their settings back`() {
        stubUser()
        val settings = PhotographerSettings(userId = photographerId, verificationStatus = VerificationStatus.APPROVED)
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(Optional.of(settings))

        assertSame(settings, guard.requireApproved(photographerId, "uploading"))
    }

    @Test
    fun `a suspended photographer is refused with ACCOUNT_SUSPENDED`() {
        stubUser(suspended = true)

        val ex = assertFailsWith<ApiException> { guard.requireApproved(photographerId, "creating coupons") }

        assertEquals(HttpStatus.FORBIDDEN, ex.status)
        assertEquals(ErrorCodes.ACCOUNT_SUSPENDED, ex.code)
    }

    @Test
    fun `a pending photographer is refused with PHOTOGRAPHER_NOT_VERIFIED`() {
        stubUser()
        stubSettings(VerificationStatus.PENDING)

        val ex = assertFailsWith<ApiException> { guard.requireApproved(photographerId, "creating an event") }

        assertEquals(HttpStatus.FORBIDDEN, ex.status)
        assertEquals(ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED, ex.code)
    }

    @Test
    fun `a photographer with no settings row is refused (fail-closed)`() {
        stubUser()
        stubSettings(null)

        val ex = assertFailsWith<ApiException> { guard.requireApproved(photographerId, "uploading") }

        assertEquals(ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED, ex.code)
    }

    @Test
    fun `a missing user row is a 404`() {
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.empty())

        val ex = assertFailsWith<NotFoundException> { guard.requireApproved(photographerId, "uploading") }

        assertEquals(ErrorCodes.USER_NOT_FOUND, ex.code)
    }
}
