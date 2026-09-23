package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.PublicProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.orders.CouponService
import com.quickpitik.service.profile.UserDtoMapper
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageImpl
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// A suspended or not-yet-approved photographer must be invisible on the public
// site — 404, the same answer as an unknown handle, so existence isn't leaked.
// The guard is copied into both public reads and neither copy had a test
// (2026-09-11 test-suite audit).
class PublicPhotographerServiceTest {

    private lateinit var photographerSettingsRepository: PhotographerSettingsRepository
    private lateinit var userRepository: UserRepository
    private lateinit var eventPhotographerRepository: EventPhotographerRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var photoRepository: PhotoRepository
    private lateinit var storageService: StorageService

    @BeforeEach
    fun setUp() {
        photographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        eventPhotographerRepository = Mockito.mock(EventPhotographerRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
    }

    private fun service() = PublicPhotographerService(
        photographerSettingsRepository,
        userRepository,
        eventPhotographerRepository,
        eventRepository,
        photoRepository,
        StorageProperties(),
        storageService,
        Mockito.mock(CouponService::class.java),
        PublicProperties(),
        Mockito.mock(EventDtoMapper::class.java),
        Mockito.mock(UserDtoMapper::class.java),
    )

    private fun photographer(handle: String, status: VerificationStatus, suspended: Boolean) {
        val userId = UUID.randomUUID()
        Mockito.`when`(photographerSettingsRepository.findByHandleIgnoreCase(handle))
            .thenReturn(PhotographerSettings(userId = userId, handle = handle, verificationStatus = status))
        val user = User(id = userId, email = "$handle@test.local", passwordHash = "x", name = "Shooter", role = Role.PHOTOGRAPHER)
        if (suspended) user.suspendedAt = OffsetDateTime.now()
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(user))
    }

    @Test
    fun `public gallery previews use derivatives or null and never the original`() {
        photographer("shooter", VerificationStatus.APPROVED, suspended = false)
        val ownerId = photographerSettingsRepository.findByHandleIgnoreCase("shooter")!!.userId
        val event = Event(slug = "race", name = "Race", date = LocalDate.of(2026, 9, 1), location = "Cebu", status = EventStatus.ACTIVE)
        Mockito.`when`(eventRepository.findBySlugAndDeletedAtIsNull("race")).thenReturn(event)
        val photos = listOf(
            Photo(eventId = event.id, photographerId = ownerId, s3Key = "original.jpg", pricePhp = BigDecimal.TEN),
            Photo(eventId = event.id, photographerId = ownerId, s3Key = "original.jpg", pricePhp = BigDecimal.TEN, thumbnailS3Key = "thumbnail.jpg"),
            Photo(eventId = event.id, photographerId = ownerId, s3Key = "original.jpg", pricePhp = BigDecimal.TEN, thumbnailS3Key = "thumbnail.jpg", watermarkS3Key = "watermark.jpg"),
        )
        Mockito.`when`(photoRepository.findPhotographerLibraryByStatus(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(PageImpl(photos))
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg()))
            .thenAnswer { "https://storage/${it.getArgument<String>(0)}" }

        val items = service().listEventPhotos("shooter", "race", PaginationParams.of(0, 20)).items

        assertEquals(listOf(null, "https://storage/thumbnail.jpg", "https://storage/watermark.jpg"), items.map { it.imageUrl })
        assertTrue(items.all { it.cleanUrl == null && it.downloadUrl == null })
        Mockito.verify(storageService, Mockito.never()).presignedGetUrl(eqArg("original.jpg"), anyArg())
    }

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value

    @Test
    fun `a suspended photographer profile is not found`() {
        photographer("shooter", VerificationStatus.APPROVED, suspended = true)

        val ex = assertFailsWith<NotFoundException> { service().getProfile("Shooter") }

        assertEquals(ErrorCodes.NOT_FOUND, ex.code)
        Mockito.verifyNoInteractions(eventPhotographerRepository)
    }

    @Test
    fun `an unverified photographer gallery is not found`() {
        photographer("pending", VerificationStatus.PENDING, suspended = false)

        val ex = assertFailsWith<NotFoundException> {
            service().listEventPhotos("pending", "any-race", PaginationParams(offset = 0, limit = 20))
        }

        assertEquals(ErrorCodes.NOT_FOUND, ex.code)
        Mockito.verifyNoInteractions(eventRepository)
        Mockito.verifyNoInteractions(photoRepository)
    }
}
