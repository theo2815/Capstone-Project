package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.PhotographerMessage
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ValidationException
import com.quickpitik.exception.ApiException
import org.springframework.http.HttpStatus
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
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// Suspension is only as strong as its session kill. The access token's
// `suspended` claim is a backstop with a 15-minute TTL; what actually ends a
// suspended user's access is revoking every refresh token so they can never
// mint another access token. Nothing pinned that call, nor the guard that keeps
// one admin from suspending another (2026-09-11 test-suite audit).
class AdminUserServiceSuspendTest {

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
    private val targetId = UUID.randomUUID()

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
        Mockito.mock(AdminActionCommandService::class.java),
        Mockito.mock(com.quickpitik.repository.PhotographerMessageRepository::class.java),
        Mockito.mock(com.quickpitik.repository.RunnerMessageRepository::class.java),
    )

    private fun user(role: Role): User = User(
        id = targetId,
        email = "target@test.local",
        passwordHash = "x",
        name = "Target",
        role = role,
    )

    @Test
    fun `suspend stamps the account and kills every session`() {
        val photographer = user(Role.PHOTOGRAPHER)
        Mockito.`when`(userRepository.findById(targetId)).thenReturn(Optional.of(photographer))
        Mockito.`when`(userRepository.lockById(targetId)).thenReturn(photographer)
        Mockito.`when`(photographerSettingsRepository.findById(targetId))
            .thenReturn(Optional.of(PhotographerSettings(userId = targetId, verificationStatus = VerificationStatus.APPROVED)))
        Mockito.`when`(adminDecisionLogService.logUserDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetUserId = targetId, decision = "suspended"))
        // Captured via thenAnswer rather than eq(): Mockito's matchers hand
        // Kotlin a null that the call site null-checks before the mock runs.
        var pushed: List<Any?> = emptyList()
        Mockito.`when`(adminDecisionLogService.pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull()))
            .thenAnswer {
                pushed = it.arguments.toList()
                PhotographerMessage(
                    photographerId = targetId,
                    kindWire = PhotographerMessageKind.SUSPENDED.wire,
                    body = "suspended",
                )
            }
        Mockito.`when`(regionsService.all()).thenReturn(emptyList())

        val service = service()
        service.suspend(adminId, targetId, "chargeback abuse")
        val suspendedAt = photographer.suspendedAt
        val ex = assertFailsWith<ApiException> { service.suspend(adminId, targetId, "Repeat") }
        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        assertEquals(suspendedAt, photographer.suspendedAt)
        Mockito.verify(adminDecisionLogService).logUserDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
        Mockito.verify(adminDecisionLogService).pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull())

        assertNotNull(photographer.suspendedAt, "the account must be stamped suspended")
        assertEquals("chargeback abuse", photographer.suspensionReason)
        Mockito.verify(userRepository).save(photographer)
        // The part that actually locks them out: no refresh token survives.
        Mockito.verify(refreshTokenService).revokeAllForUser(targetId)
        // The photographer is told, by this admin, with the suspended kind.
        assertEquals(targetId, pushed[0])
        assertEquals(PhotographerMessageKind.SUSPENDED, pushed[1])
        assertEquals(adminId, pushed[3])
    }

    @Test
    fun `an admin account cannot be suspended`() {
        val admin = user(Role.ADMIN)
        Mockito.`when`(userRepository.findById(targetId)).thenReturn(Optional.of(admin))
        Mockito.`when`(userRepository.lockById(targetId)).thenReturn(admin)

        val ex = assertFailsWith<ValidationException> { service().suspend(adminId, targetId, "power struggle") }

        assertEquals(ErrorCodes.VALIDATION_ERROR, ex.code)
        assertEquals("userId", ex.field)
        assertNull(admin.suspendedAt)
        Mockito.verify(userRepository, Mockito.never()).save(anyArg())
        Mockito.verifyNoInteractions(refreshTokenService)
        Mockito.verifyNoInteractions(adminDecisionLogService)
    }

    @Test
    fun `unsuspend accepts once without another message or token revocation`() {
        val photographer = user(Role.PHOTOGRAPHER).apply {
            suspendedAt = java.time.OffsetDateTime.now()
            suspensionReason = "Original reason"
        }
        Mockito.`when`(userRepository.findById(targetId)).thenReturn(Optional.of(photographer))
        Mockito.`when`(userRepository.lockById(targetId)).thenReturn(photographer)
        Mockito.`when`(adminDecisionLogService.logUserDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetUserId = targetId, decision = "unsuspended"))
        Mockito.`when`(regionsService.all()).thenReturn(emptyList())
        val service = service()
        service.unsuspend(adminId, targetId)
        val ex = assertFailsWith<ApiException> { service.unsuspend(adminId, targetId) }
        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        assertNull(photographer.suspendedAt)
        assertNull(photographer.suspensionReason)
        Mockito.verify(adminDecisionLogService).logUserDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
        Mockito.verify(adminDecisionLogService).pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull())
        Mockito.verifyNoInteractions(refreshTokenService)
    }
}
