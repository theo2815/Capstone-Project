package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.auth.RefreshTokenService
import com.quickpitik.service.photographer.PhotographerSettingsService
import com.quickpitik.service.profile.UserDtoMapper
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// R-10: an admin reject or reset holds the profile; only the admin approve releases it.
class AdminUserServiceHoldTest {
    private val adminId = UUID.randomUUID()
    private val photographerId = UUID.randomUUID()
    private val userRepository = Mockito.mock(UserRepository::class.java)
    private val settingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
    private val photographerSettingsService = Mockito.mock(PhotographerSettingsService::class.java)
    private val decisionLog = Mockito.mock(AdminDecisionLogService::class.java) { call ->
        if (call.method.name == "logUserDecision")
            AdminDecisionLog(adminId = adminId, targetUserId = photographerId, decision = call.arguments[2] as String)
        else null
    }
    private val settings = PhotographerSettings(userId = photographerId, verificationStatus = VerificationStatus.PENDING)

    init {
        Mockito.`when`(userRepository.lockById(photographerId)).thenReturn(
            User(id = photographerId, email = "p@test.local", passwordHash = "unused", name = "P", role = Role.PHOTOGRAPHER))
        Mockito.`when`(settingsRepository.findByUserIdForUpdate(photographerId)).thenReturn(settings)
        Mockito.`when`(settingsRepository.save(settings)).thenReturn(settings)
    }

    private fun service() = AdminUserService(userRepository, settingsRepository,
        Mockito.mock(SocialLinkRepository::class.java), Mockito.mock(PayoutAccountRepository::class.java),
        Mockito.mock(AdminDecisionLogRepository::class.java), Mockito.mock(RegionsService::class.java),
        decisionLog, Mockito.mock(StorageService::class.java), StorageProperties(),
        Mockito.mock(UserDtoMapper::class.java), Mockito.mock(RunnerMessagesService::class.java),
        Mockito.mock(RefreshTokenService::class.java), Mockito.mock(AdminActionCommandService::class.java),
        Mockito.mock(PhotographerMessageRepository::class.java), Mockito.mock(RunnerMessageRepository::class.java),
        photographerSettingsService)

    @Test
    fun `reject puts the profile on hold`() {
        val row = service().reject(adminId, photographerId, "Blurry ID")
        assertTrue(settings.verificationAdminHold)
        assertTrue(row.adminHold)
    }

    @Test
    fun `reset puts the profile on hold`() {
        settings.verificationStatus = VerificationStatus.APPROVED
        val row = service().resetVerification(adminId, photographerId, "Wrong ID")
        assertTrue(settings.verificationAdminHold)
        assertTrue(row.adminHold)
    }

    @Test
    fun `approving a held profile clears the hold and approves it when identity passed`() {
        settings.verificationAdminHold = true
        Mockito.`when`(photographerSettingsService.approveIfReady(photographerId)).thenAnswer {
            assertFalse(settings.verificationAdminHold, "the hold is cleared before the approval runs")
            settings.verificationStatus = VerificationStatus.APPROVED
            true
        }

        val row = service().approveHeld(adminId, photographerId, 0)

        assertFalse(row.adminHold)
        assertEquals("approved", row.verificationStatus)
        Mockito.verify(decisionLog).logUserDecision(adminId, photographerId, "approved", null, null, null)
    }

    @Test
    fun `approving a held profile whose identity has not passed leaves it pending without the hold`() {
        settings.verificationAdminHold = true
        Mockito.`when`(photographerSettingsService.approveIfReady(photographerId)).thenReturn(false)

        val row = service().approveHeld(adminId, photographerId, null)

        assertFalse(settings.verificationAdminHold)
        assertEquals("pending", row.verificationStatus)
        Mockito.verify(photographerSettingsService).approveIfReady(photographerId)
        Mockito.verify(decisionLog).logUserDecision(adminId, photographerId, "approved",
            "Hold released; goes live when the identity check passes", null, null)
    }

    @Test
    fun `the detail answer carries the hold, so the drawer needs no list row`() {
        settings.verificationAdminHold = true
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(java.util.Optional.of(
            User(id = photographerId, email = "p@test.local", passwordHash = "unused", name = "P", role = Role.PHOTOGRAPHER)))
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(java.util.Optional.of(settings))

        assertTrue(service().detail(photographerId).adminHold)
        settings.verificationAdminHold = false
        assertFalse(service().detail(photographerId).adminHold)
    }

    @Test
    fun `approve on a profile that is not held is a conflict`() {
        val ex = assertFailsWith<ApiException> { service().approveHeld(adminId, photographerId, null) }
        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.CONFLICT, ex.code)
        Mockito.verifyNoInteractions(photographerSettingsService)
        Mockito.verify(decisionLog, Mockito.never()).logUserDecision(adminId, photographerId, "approved", null, null, null)
    }

    @Test
    fun `approve on a held profile that is not pending is a conflict`() {
        settings.verificationAdminHold = true
        settings.verificationStatus = VerificationStatus.INCOMPLETE
        val ex = assertFailsWith<ApiException> { service().approveHeld(adminId, photographerId, null) }
        assertEquals(ErrorCodes.CONFLICT, ex.code)
        assertTrue(settings.verificationAdminHold)
        Mockito.verifyNoInteractions(photographerSettingsService)
    }

    @Test
    fun `a stale revision is refused`() {
        settings.verificationAdminHold = true
        val ex = assertFailsWith<ApiException> { service().approveHeld(adminId, photographerId, 7) }
        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertTrue(settings.verificationAdminHold)
        Mockito.verifyNoInteractions(photographerSettingsService)
    }
}
