package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.HandlePatchRequest
import com.quickpitik.dto.photographer.MissingRequirement
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.ReservedHandleRepository
import com.quickpitik.repository.SocialLinkRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.service.storage.StorageService
import org.hibernate.exception.ConstraintViolationException
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.http.HttpStatus
import java.sql.SQLException
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// Two behaviours locked here:
//
//  1. Handle claiming is a check-then-save, so two photographers racing for the
//     same handle both pass the in-memory check. The UNIQUE index is the real
//     gate — but a raw DataIntegrityViolationException has no handler and used
//     to surface as a 500. It must translate to 409 HANDLE_TAKEN.
//  2. Cover + watermark uploads must reject oversized bytes BEFORE ImageIO
//     decodes them.
class PhotographerSettingsServiceTest {

    private lateinit var photographerSettingsRepository: PhotographerSettingsRepository
    private lateinit var userRepository: UserRepository
    private lateinit var reservedHandleRepository: ReservedHandleRepository
    private lateinit var regionsService: RegionsService
    private lateinit var storageService: StorageService
    private lateinit var socialLinkRepository: SocialLinkRepository
    private lateinit var payoutAccountRepository: PayoutAccountRepository
    private lateinit var eventPublisher: ApplicationEventPublisher

    private val userId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        photographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        reservedHandleRepository = Mockito.mock(ReservedHandleRepository::class.java)
        regionsService = Mockito.mock(RegionsService::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        socialLinkRepository = Mockito.mock(SocialLinkRepository::class.java)
        payoutAccountRepository = Mockito.mock(PayoutAccountRepository::class.java)
        eventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)
    }

    private fun service() = PhotographerSettingsService(
        photographerSettingsRepository,
        userRepository,
        reservedHandleRepository,
        regionsService,
        storageService,
        StorageProperties(),
        socialLinkRepository,
        payoutAccountRepository,
        eventPublisher,
    )

    // A Spring DataIntegrityViolationException wrapping a Hibernate
    // ConstraintViolationException that names the given constraint — the shape a
    // real Postgres unique violation arrives in.
    private fun dataIntegrityViolation(constraintName: String): DataIntegrityViolationException =
        DataIntegrityViolationException(
            "constraint violation",
            ConstraintViolationException("constraint violation", SQLException("violation"), constraintName),
        )

    private fun stubFreeHandle() {
        Mockito.`when`(reservedHandleRepository.existsByHandle(anyArg())).thenReturn(false)
        Mockito.`when`(photographerSettingsRepository.findById(userId))
            .thenReturn(Optional.of(PhotographerSettings(userId = userId)))
        // Fast-path uniqueness check finds nothing — the race is what follows.
        Mockito.`when`(photographerSettingsRepository.findByHandleIgnoreCase(anyArg())).thenReturn(null)
    }

    @Test
    fun `losing the handle race returns 409 HANDLE_TAKEN instead of a 500`() {
        stubFreeHandle()
        Mockito.`when`(photographerSettingsRepository.saveAndFlush(anyArg<PhotographerSettings>()))
            .thenThrow(dataIntegrityViolation("photographer_settings_handle_key"))

        val ex = assertFailsWith<ApiException> {
            service().putHandle(userId, HandlePatchRequest(handle = "juandc"))
        }

        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.HANDLE_TAKEN, ex.code)
        assertEquals("handle", ex.field)
    }

    @Test
    fun `a violation naming a different constraint is rethrown, not mislabeled as handle-taken`() {
        stubFreeHandle()
        Mockito.`when`(photographerSettingsRepository.saveAndFlush(anyArg<PhotographerSettings>()))
            .thenThrow(dataIntegrityViolation("some_future_constraint"))

        assertFailsWith<DataIntegrityViolationException> {
            service().putHandle(userId, HandlePatchRequest(handle = "juandc"))
        }
    }

    @Test
    fun `reserved handles are still rejected before any save`() {
        Mockito.`when`(reservedHandleRepository.existsByHandle("admin")).thenReturn(true)

        val ex = assertFailsWith<ApiException> {
            service().putHandle(userId, HandlePatchRequest(handle = "Admin"))
        }

        assertEquals(ErrorCodes.RESERVED_HANDLE, ex.code)
        Mockito.verify(photographerSettingsRepository, Mockito.never()).saveAndFlush(anyArg<PhotographerSettings>())
    }

    @Test
    fun `cover over 10MB is rejected before decode`() {
        val ex = assertFailsWith<ApiException> {
            service().uploadCover(userId, ByteArray(10 * 1024 * 1024 + 1), "image/jpeg")
        }

        assertEquals(HttpStatus.PAYLOAD_TOO_LARGE, ex.status)
        assertEquals(ErrorCodes.PAYLOAD_TOO_LARGE, ex.code)
        Mockito.verifyNoInteractions(storageService)
    }

    @Test
    fun `watermark over 10MB is rejected before decode`() {
        val ex = assertFailsWith<ApiException> {
            service().uploadWatermark(userId, ByteArray(10 * 1024 * 1024 + 1), "image/png")
        }

        assertEquals(HttpStatus.PAYLOAD_TOO_LARGE, ex.status)
        assertEquals(ErrorCodes.PAYLOAD_TOO_LARGE, ex.code)
        Mockito.verifyNoInteractions(storageService)
    }

    // ─── collectMissing — the single required-field predicate both clients
    // and the admin-approve gate consume. Pins each of the 8 items so a future
    // edit to the set can't silently drift from what the FE guards on.

    private fun completeSettings() = PhotographerSettings(
        userId = userId,
        handle = "aira",
        brandName = "Aira Studio",
        coverS3Key = "covers/$userId.jpg",
        watermarkS3Key = "watermarks/$userId.png",
        regionCode = "07",
        provinceCode = "0722",
        verificationStatus = VerificationStatus.APPROVED,
    )

    private fun userNoAvatar() =
        User(id = userId, email = "p@test.local", passwordHash = "x", name = "Aira", role = Role.PHOTOGRAPHER)

    @Test
    fun `collectMissing is empty when every required field is set`() {
        Mockito.`when`(userRepository.findById(userId))
            .thenReturn(Optional.of(userNoAvatar().apply { avatarS3Key = "avatars/$userId.jpg" }))
        Mockito.`when`(socialLinkRepository.countByUserId(userId)).thenReturn(1L)
        Mockito.`when`(payoutAccountRepository.countByUserId(userId)).thenReturn(1L)

        val result = service().collectMissing(userId, completeSettings())

        assertTrue(result.isComplete)
        assertEquals(emptyList(), result.missing)
    }

    @Test
    fun `collectMissing reports all eight requirements in order for a blank profile`() {
        // No avatar on the user row, empty settings; an unstubbed Long mock
        // returns 0, so socials/payouts read as absent.
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(userNoAvatar()))

        val result = service().collectMissing(userId, PhotographerSettings(userId = userId))

        assertEquals(
            listOf(
                MissingRequirement.PROFILE_PHOTO,
                MissingRequirement.COVER_PHOTO,
                MissingRequirement.BRAND_NAME,
                MissingRequirement.WATERMARK,
                MissingRequirement.PUBLIC_HANDLE,
                MissingRequirement.REGION,
                MissingRequirement.SOCIAL_LINK,
                MissingRequirement.PAYOUT_ACCOUNT,
            ),
            result.missing,
        )
        // Labels stay verbatim — the wire contract older clients still display.
        assertEquals("profile picture", result.labels.first())
        assertEquals("payout account", result.labels.last())
    }

    @Test
    fun `collectMissing accepts a profile photo set via avatarUrl alone`() {
        // The bug class: the photo lives on the user row via avatarUrl even when
        // avatarS3Key is null — the profile-photo requirement must be satisfied.
        Mockito.`when`(userRepository.findById(userId))
            .thenReturn(Optional.of(userNoAvatar().apply { avatarUrl = "https://cdn/avatar.jpg" }))
        Mockito.`when`(socialLinkRepository.countByUserId(userId)).thenReturn(1L)
        Mockito.`when`(payoutAccountRepository.countByUserId(userId)).thenReturn(1L)

        val result = service().collectMissing(userId, completeSettings())

        assertTrue(result.isComplete)
    }
}
