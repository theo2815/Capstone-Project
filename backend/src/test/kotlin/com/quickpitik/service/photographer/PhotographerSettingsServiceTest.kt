package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.HandlePatchRequest
import com.quickpitik.dto.photographer.MissingRequirement
import com.quickpitik.dto.photographer.WatermarkSettingsRequest
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.ReservedHandleRepository
import com.quickpitik.repository.SocialLinkRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.service.image.ImagePixelGuardTest.Companion.overBudgetPngHeader
import com.quickpitik.service.storage.StorageService
import org.hibernate.exception.ConstraintViolationException
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.http.HttpStatus
import org.springframework.http.converter.json.Jackson2ObjectMapperBuilder
import java.sql.SQLException
import java.util.Base64
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

    @Test
    fun `truncated WebP cover is rejected before storage or settings writes`() {
        val bytes = Base64.getDecoder().decode("UklGRjIAAABXRUJQVlA4ICYAAAA=")

        val ex = assertFailsWith<ValidationException> {
            service().uploadCover(userId, bytes, "image/webp")
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, ex.code)
        assertEquals("file", ex.field)
        Mockito.verifyNoInteractions(storageService, photographerSettingsRepository)
    }

    @Test
    fun `truncated WebP watermark is rejected before storage or settings writes`() {
        val bytes = Base64.getDecoder().decode("UklGRjIAAABXRUJQVlA4ICYAAAA=")

        val ex = assertFailsWith<ValidationException> {
            service().uploadWatermark(userId, bytes, "image/webp")
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, ex.code)
        assertEquals("file", ex.field)
        Mockito.verifyNoInteractions(storageService, photographerSettingsRepository)
    }

    @Test
    fun `cover dimensions over 80MP are rejected before storage`() {
        val ex = assertFailsWith<ValidationException> {
            service().uploadCover(userId, overBudgetPngHeader(), "image/png")
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, ex.code)
        assertEquals("Image dimensions exceed the supported maximum.", ex.message)
        assertEquals("file", ex.field)
        Mockito.verifyNoInteractions(storageService)
    }

    @Test
    fun `watermark dimensions over 80MP are rejected before storage`() {
        val ex = assertFailsWith<ValidationException> {
            service().uploadWatermark(userId, overBudgetPngHeader(), "image/png")
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, ex.code)
        assertEquals("Image dimensions exceed the supported maximum.", ex.message)
        assertEquals("file", ex.field)
        Mockito.verifyNoInteractions(storageService)
    }

    @Test
    fun `non-finite watermark fields are rejected before reading or changing settings`() {
        val mapper = Jackson2ObjectMapperBuilder.json().build<com.fasterxml.jackson.databind.ObjectMapper>()
        val fields = listOf("opacity", "scale", "landscapeX", "landscapeY", "portraitX", "portraitY")
        for (field in fields) for (value in listOf("NaN", "Infinity", "-Infinity")) {
            // Jackson accepts these quoted strings as Doubles on the real API.
            val body = fields.joinToString(prefix = "{", postfix = "}") {
                "\"$it\":" + if (it == field) "\"$value\"" else "0.3"
            }
            val req = mapper.readValue(body, WatermarkSettingsRequest::class.java)
            val ex = assertFailsWith<ValidationException>("$field=$value") {
                service().putWatermarkSettings(userId, req)
            }
            assertEquals(ErrorCodes.VALIDATION_ERROR, ex.code)
            assertEquals(field, ex.field)
        }
        Mockito.verifyNoInteractions(photographerSettingsRepository, userRepository)
    }

    @Test
    fun `finite watermark values still clamp and round-trip through brand details`() {
        val settings = PhotographerSettings(userId = userId)
        Mockito.`when`(photographerSettingsRepository.findById(userId)).thenReturn(Optional.of(settings))
        Mockito.`when`(photographerSettingsRepository.save(anyArg<PhotographerSettings>()))
            .thenAnswer { it.getArgument<PhotographerSettings>(0) }
        val service = service()
        service.putWatermarkSettings(userId, WatermarkSettingsRequest(2.0, -1.0, -5.0, 5.0, 0.3, 0.7))

        assertEquals(
            mapOf("opacity" to 1.0, "scale" to 0.05, "landscapeX" to 0.0,
                "landscapeY" to 1.0, "portraitX" to 0.3, "portraitY" to 0.7),
            service.getBrandDetails(userId)["watermarkSettings"],
        )
        Mockito.verify(photographerSettingsRepository).save(settings)
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
        Mockito.`when`(payoutAccountRepository.countByUserIdAndQrS3KeyIsNotNull(userId)).thenReturn(1L)

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
        Mockito.`when`(payoutAccountRepository.countByUserIdAndQrS3KeyIsNotNull(userId)).thenReturn(1L)

        val result = service().collectMissing(userId, completeSettings())

        assertTrue(result.isComplete)
    }

    private fun stubVerification(settings: PhotographerSettings, user: User = userNoAvatar().apply {
        avatarUrl = "https://example.test/avatar.jpg"
    }) {
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(user))
        Mockito.`when`(userRepository.lockById(userId)).thenReturn(user)
        Mockito.`when`(photographerSettingsRepository.findById(userId)).thenReturn(Optional.of(settings))
        Mockito.`when`(photographerSettingsRepository.findByUserIdForUpdate(userId)).thenReturn(settings)
        Mockito.`when`(photographerSettingsRepository.save(anyArg<PhotographerSettings>())).thenAnswer { it.arguments[0] }
        Mockito.`when`(socialLinkRepository.countByUserId(userId)).thenReturn(1L)
        Mockito.`when`(payoutAccountRepository.countByUserIdAndQrS3KeyIsNotNull(userId)).thenReturn(1L)
    }

    @Test
    fun `an approved photographer can deliberately submit a new review generation`() {
        val settings = completeSettings()
        stubVerification(settings)
        assertEquals("pending", service().submitVerification(userId).status)
        assertEquals(VerificationStatus.PENDING, settings.verificationStatus)
        assertEquals(1L, settings.verificationRevision)
        Mockito.verify(eventPublisher).publishEvent(anyArg<Any>())
    }

    @Test
    fun `repeated pending submission does not create another generation or notification`() {
        val settings = completeSettings().apply { verificationStatus = VerificationStatus.PENDING }
        stubVerification(settings)
        val service = service()
        assertEquals("pending", service.submitVerification(userId).status)
        assertEquals("pending", service.submitVerification(userId).status)
        Mockito.verifyNoInteractions(eventPublisher)
        assertEquals(0L, settings.verificationRevision)
    }

    @Test
    fun `withdraw advances once from pending or approved and repeated withdrawal is a no-op`() {
        for (state in listOf(VerificationStatus.PENDING, VerificationStatus.APPROVED)) {
            val settings = completeSettings().apply { verificationStatus = state }
            stubVerification(settings)
            val service = service()
            assertEquals("incomplete", service.withdrawVerification(userId).status)
            assertEquals(1L, settings.verificationRevision)
            assertEquals("incomplete", service.withdrawVerification(userId).status)
            assertEquals(1L, settings.verificationRevision)
        }
        Mockito.verifyNoInteractions(eventPublisher)
    }

    @Test
    fun `incomplete submission preserves stored state and generation`() {
        val settings = completeSettings().apply { watermarkS3Key = null }
        stubVerification(settings)
        val result = service().submitVerification(userId)
        assertEquals("incomplete", result.status)
        assertEquals(listOf("WATERMARK"), result.missingCodes)
        assertEquals(VerificationStatus.APPROVED, settings.verificationStatus)
        assertEquals(0L, settings.verificationRevision)
        Mockito.verifyNoInteractions(eventPublisher)
    }

    @Test
    fun `verification writers refuse a wrong role even if settings exist`() {
        stubVerification(completeSettings(), userNoAvatar().apply { role = Role.RUNNER; avatarUrl = "https://example.test/avatar.jpg" })
        assertEquals(ErrorCodes.FORBIDDEN, assertFailsWith<ConflictException> { service().submitVerification(userId) }.code)
        assertEquals(ErrorCodes.FORBIDDEN, assertFailsWith<ConflictException> { service().withdrawVerification(userId) }.code)
        Mockito.verifyNoInteractions(eventPublisher)
    }

    @Test
    fun `resubmission does not reinstate a suspended photographer`() {
        val settings = completeSettings()
        val user = userNoAvatar().apply {
            avatarUrl = "https://example.test/avatar.jpg"
            suspendedAt = java.time.OffsetDateTime.now()
            suspensionReason = "Original suspension"
        }
        stubVerification(settings, user)
        val result = service().submitVerification(userId)
        assertEquals("pending", result.status)
        assertEquals(user.suspendedAt.toString(), result.suspendedAt)
        assertEquals("Original suspension", result.suspensionReason)
        assertEquals(1L, settings.verificationRevision)
    }
}
