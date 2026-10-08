package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.IncompleteFields
import com.quickpitik.dto.photographer.MissingRequirement
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.PhotographerMessage
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.SocialLinkRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.auth.RefreshTokenService
import com.quickpitik.service.photographer.PhotographerSettingsService
import com.quickpitik.service.profile.UserDtoMapper
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Admin approval must enforce the SAME required-field set as the photographer's
// own submit gate. Without the gate an admin could flip an empty profile to
// APPROVED — including one whose settings row was lazy-created by
// requirePhotographerSettings on this very call — and the public profile would
// then render broken.
class AdminUserServiceApproveTest {

    private lateinit var userRepository: UserRepository
    private lateinit var photographerSettingsRepository: PhotographerSettingsRepository
    private lateinit var socialLinkRepository: SocialLinkRepository
    private lateinit var payoutAccountRepository: PayoutAccountRepository
    private lateinit var adminDecisionLogRepository: AdminDecisionLogRepository
    private lateinit var regionsService: RegionsService
    private lateinit var adminDecisionLogService: AdminDecisionLogService
    private lateinit var storageService: StorageService
    private lateinit var userDtoMapper: UserDtoMapper
    private lateinit var runnerMessagesService: RunnerMessagesService
    private lateinit var photographerSettingsService: PhotographerSettingsService
    private lateinit var refreshTokenService: RefreshTokenService

    private val adminId = UUID.randomUUID()
    private val photographerId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        userRepository = Mockito.mock(UserRepository::class.java)
        photographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        socialLinkRepository = Mockito.mock(SocialLinkRepository::class.java)
        payoutAccountRepository = Mockito.mock(PayoutAccountRepository::class.java)
        adminDecisionLogRepository = Mockito.mock(AdminDecisionLogRepository::class.java)
        regionsService = Mockito.mock(RegionsService::class.java)
        adminDecisionLogService = Mockito.mock(AdminDecisionLogService::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        userDtoMapper = Mockito.mock(UserDtoMapper::class.java)
        runnerMessagesService = Mockito.mock(RunnerMessagesService::class.java)
        photographerSettingsService = Mockito.mock(PhotographerSettingsService::class.java)
        refreshTokenService = Mockito.mock(RefreshTokenService::class.java)
    }

    private fun service() = AdminUserService(
        userRepository,
        photographerSettingsRepository,
        socialLinkRepository,
        payoutAccountRepository,
        adminDecisionLogRepository,
        regionsService,
        adminDecisionLogService,
        storageService,
        StorageProperties(),
        userDtoMapper,
        runnerMessagesService,
        photographerSettingsService,
        refreshTokenService,
    )

    private fun photographer(): User = User(
        id = photographerId,
        email = "p@test.local",
        passwordHash = "x",
        name = "Photog",
        role = Role.PHOTOGRAPHER,
    )

    private fun settings() = PhotographerSettings(
        userId = photographerId,
        verificationStatus = VerificationStatus.PENDING,
    )

    private fun stubLookups(s: PhotographerSettings) {
        val user = photographer()
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(user))
        Mockito.`when`(userRepository.lockById(photographerId)).thenReturn(user)
        Mockito.`when`(photographerSettingsRepository.findById(photographerId)).thenReturn(Optional.of(s))
        Mockito.`when`(photographerSettingsRepository.findByUserIdForUpdate(photographerId)).thenReturn(s)
    }

    @Test
    fun `approve refuses an incomplete profile and leaves the status untouched`() {
        val s = settings()
        stubLookups(s)
        Mockito.`when`(photographerSettingsService.collectMissing(anyArg(), anyArg()))
            .thenReturn(IncompleteFields(missing = listOf(MissingRequirement.WATERMARK, MissingRequirement.PAYOUT_ACCOUNT)))

        val ex = assertFailsWith<ApiException> { service().approve(adminId, photographerId) }

        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, ex.status)
        assertEquals(ErrorCodes.INCOMPLETE_PROFILE, ex.code)
        // The photographer must still be PENDING — no half-applied approval.
        assertEquals(VerificationStatus.PENDING, s.verificationStatus)
        Mockito.verify(photographerSettingsRepository, Mockito.never()).save(anyArg())
        // No inbox message either: the photographer must not be told they're live.
        Mockito.verify(adminDecisionLogService, Mockito.never())
            .pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull())
    }

    @Test
    fun `approve flips to APPROVED when nothing is missing`() {
        val s = settings()
        stubLookups(s)
        Mockito.`when`(photographerSettingsService.collectMissing(anyArg(), anyArg()))
            .thenReturn(IncompleteFields(missing = emptyList()))
        Mockito.`when`(adminDecisionLogService.logUserDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetUserId = photographerId, decision = "approved"))
        Mockito.`when`(
            adminDecisionLogService.pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull()),
        ).thenReturn(
            PhotographerMessage(
                photographerId = photographerId,
                kindWire = PhotographerMessageKind.VERIFICATION_APPROVED.wire,
                body = "approved",
            ),
        )
        Mockito.`when`(regionsService.all()).thenReturn(emptyList())

        service().approve(adminId, photographerId)

        assertEquals(VerificationStatus.APPROVED, s.verificationStatus)
    }

    // P1.6 — the admin review pane is a READ. requirePhotographerSettings used to
    // lazy-create the settings row here under readOnly = true, where the INSERT is
    // never flushed: a wasted SELECT per call and the exact pitfall logged in vault
    // backend/notes/jpa-pitfalls.md (2026-05-25).
    @Test
    fun `photographerSettings renders defaults without creating a settings row`() {
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(photographer()))
        Mockito.`when`(photographerSettingsRepository.findById(photographerId)).thenReturn(Optional.empty())
        Mockito.`when`(socialLinkRepository.findAllByUserIdOrderByCreatedAtAsc(photographerId)).thenReturn(emptyList())
        Mockito.`when`(payoutAccountRepository.findAllByUserIdOrderByCreatedAtAsc(photographerId)).thenReturn(emptyList())
        // Stubbed so a regressed build reaches the verify below instead of NPEing on
        // an unstubbed save() — the failure must name the persist, not a null.
        Mockito.`when`(photographerSettingsRepository.save(anyArg())).thenAnswer { it.arguments[0] }

        val dto = service().photographerSettings(photographerId)

        Mockito.verify(photographerSettingsRepository, Mockito.never()).save(anyArg())
        // The pane still renders: the caller gets an unsaved row carrying the defaults.
        assertEquals(photographerId.toString(), dto.userId)
        assertEquals("none", dto.brandColor)
        assertEquals("", dto.bio)
    }

    private fun stubDecision(s: PhotographerSettings) {
        stubLookups(s)
        Mockito.`when`(photographerSettingsService.collectMissing(anyArg(), anyArg()))
            .thenReturn(IncompleteFields(missing = emptyList()))
        Mockito.`when`(regionsService.all()).thenReturn(emptyList())
        Mockito.`when`(adminDecisionLogService.logUserDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetUserId = photographerId, decision = "test"))
    }

    @Test
    fun `a repeated approval conflicts without another decision or message`() {
        val s = settings()
        stubDecision(s)
        val service = service()
        service.approve(adminId, photographerId)
        val ex = assertFailsWith<ApiException> { service.approve(adminId, photographerId) }
        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        assertEquals(VerificationStatus.APPROVED, s.verificationStatus)
        Mockito.verify(adminDecisionLogService).logUserDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
        Mockito.verify(adminDecisionLogService).pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull())
    }

    @Test
    fun `a repeated rejection conflicts without another decision or message`() {
        val s = settings()
        stubDecision(s)
        val service = service()
        service.reject(adminId, photographerId, "Please correct the evidence")
        val ex = assertFailsWith<ApiException> { service.reject(adminId, photographerId, "Another reason") }
        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        assertEquals(VerificationStatus.INCOMPLETE, s.verificationStatus)
        Mockito.verify(adminDecisionLogService).logUserDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
        Mockito.verify(adminDecisionLogService).pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull())
    }

    @Test
    fun `a repeated reset conflicts without another decision or message`() {
        val s = settings().apply { verificationStatus = VerificationStatus.APPROVED }
        stubDecision(s)
        val service = service()
        service.resetVerification(adminId, photographerId, "Please review your profile")
        val ex = assertFailsWith<ApiException> { service.resetVerification(adminId, photographerId, "Again") }
        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        assertEquals(VerificationStatus.INCOMPLETE, s.verificationStatus)
        Mockito.verify(adminDecisionLogService).logUserDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
        Mockito.verify(adminDecisionLogService).pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull())
    }

    @Test
    fun `approval and rejection refuse every non-pending state`() {
        for (state in listOf(VerificationStatus.INCOMPLETE, VerificationStatus.APPROVED, VerificationStatus.REJECTED)) {
            val s = settings().apply { verificationStatus = state }
            stubDecision(s)
            assertEquals(ErrorCodes.INVALID_STATE_TRANSITION,
                assertFailsWith<ApiException> { service().approve(adminId, photographerId) }.code)
            assertEquals(ErrorCodes.INVALID_STATE_TRANSITION,
                assertFailsWith<ApiException> { service().reject(adminId, photographerId, "Reason") }.code)
            assertEquals(state, s.verificationStatus)
        }
        Mockito.verifyNoInteractions(adminDecisionLogService)
    }

    @Test
    fun `reset accepts pending approved and legacy rejected`() {
        for (state in listOf(VerificationStatus.PENDING, VerificationStatus.APPROVED, VerificationStatus.REJECTED)) {
            val s = settings().apply { verificationStatus = state }
            stubDecision(s)
            assertEquals("incomplete", service().resetVerification(adminId, photographerId, "Reason").verificationStatus)
        }
        Mockito.verify(adminDecisionLogService, Mockito.times(3)).logUserDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `verification actions refuse a runner without creating settings`() {
        val runner = photographer().apply { role = Role.RUNNER }
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(runner))
        Mockito.`when`(userRepository.lockById(photographerId)).thenReturn(runner)
        assertFailsWith<ValidationException> { service().approve(adminId, photographerId) }
        assertFailsWith<ValidationException> { service().reject(adminId, photographerId, "Reason") }
        assertFailsWith<ValidationException> { service().resetVerification(adminId, photographerId, "Reason") }
        Mockito.verify(photographerSettingsRepository, Mockito.never()).save(anyArg())
        Mockito.verifyNoInteractions(adminDecisionLogService)
    }
}
