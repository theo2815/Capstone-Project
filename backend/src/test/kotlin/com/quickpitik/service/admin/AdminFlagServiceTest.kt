package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Flag
import com.quickpitik.entity.FlagStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.anyString
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.math.BigDecimal
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

// The flag queue mutates photos (hide → HIDDEN, dismiss/resolve of a hidden
// flag → LIVE again). These pin the transition table and the cascade so a
// mis-click can't silently strand a photo, and hydrate's fallbacks so the
// admin card never renders a null handle.
class AdminFlagServiceTest {

    private lateinit var flagRepository: FlagRepository
    private lateinit var photoRepository: PhotoRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var userRepository: UserRepository
    private lateinit var photographerSettingsRepository: PhotographerSettingsRepository
    private lateinit var storageService: StorageService

    private val adminId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        flagRepository = Mockito.mock(FlagRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        photographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        Mockito.`when`(eventRepository.findAllById(anyArg())).thenReturn(emptyList())
        Mockito.`when`(userRepository.findAllById(anyArg())).thenReturn(emptyList())
        Mockito.`when`(photographerSettingsRepository.findAllById(anyArg())).thenReturn(emptyList())
        Mockito.`when`(photoRepository.findAllById(anyArg())).thenReturn(emptyList())
    }

    private fun service() = AdminFlagService(
        flagRepository,
        photoRepository,
        eventRepository,
        userRepository,
        photographerSettingsRepository,
        storageService,
        StorageProperties(),
        AdminProperties(flagsEnabled = true),
    )

    private fun photo(status: PhotoStatus = PhotoStatus.LIVE) = Photo(
        eventId = UUID.randomUUID(),
        s3Key = "events/e/photos/p/original.jpg",
        pricePhp = BigDecimal.TEN,
        status = status,
    )

    private fun flag(status: FlagStatus, photo: Photo? = null): Flag {
        val f = Flag(
            targetKindWire = FlagTargetKind.PHOTO.wire,
            targetId = photo?.id ?: UUID.randomUUID(),
            reason = "inappropriate",
            statusWire = status.wire,
        )
        Mockito.`when`(flagRepository.findById(f.id)).thenReturn(Optional.of(f))
        if (photo != null) {
            Mockito.`when`(photoRepository.findById(photo.id)).thenReturn(Optional.of(photo))
            Mockito.`when`(photoRepository.findAllById(anyArg())).thenReturn(listOf(photo))
        }
        return f
    }

    @Test
    fun `escalate moves an open flag without stamping a resolution`() {
        val f = flag(FlagStatus.OPEN)
        val dto = service().escalate(adminId, f.id, "needs a second look")
        assertEquals("escalated", dto.status)
        assertNull(f.resolvedAt)
        assertNull(f.resolvedBy)
        assertEquals("needs a second look", dto.reviewerNote)
    }

    @Test
    fun `hide cascades HIDDEN onto the photo`() {
        val p = photo()
        val f = flag(FlagStatus.OPEN, p)
        val dto = service().hide(adminId, f.id, null)
        assertEquals("hidden", dto.status)
        assertEquals(PhotoStatus.HIDDEN, p.status)
        Mockito.verify(photoRepository).save(p)
    }

    @Test
    fun `hide of a closed flag is a 409`() {
        val f = flag(FlagStatus.DISMISSED)
        val ex = assertFailsWith<ApiException> { service().hide(adminId, f.id, null) }
        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        Mockito.verify(flagRepository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `dismissing a hidden flag restores the photo when nothing else hides it`() {
        val p = photo(PhotoStatus.HIDDEN)
        val f = flag(FlagStatus.HIDDEN, p)
        Mockito.`when`(
            flagRepository.existsByTargetKindWireAndTargetIdAndStatusWireAndIdNot(
                f.targetKindWire, f.targetId, FlagStatus.HIDDEN.wire, f.id,
            ),
        ).thenReturn(false)
        val dto = service().dismiss(adminId, f.id, "false alarm")
        assertEquals("dismissed", dto.status)
        assertEquals(PhotoStatus.LIVE, p.status)
    }

    @Test
    fun `dismissing a hidden flag keeps the photo hidden while another hidden flag remains`() {
        val p = photo(PhotoStatus.HIDDEN)
        val f = flag(FlagStatus.HIDDEN, p)
        Mockito.`when`(
            flagRepository.existsByTargetKindWireAndTargetIdAndStatusWireAndIdNot(
                f.targetKindWire, f.targetId, FlagStatus.HIDDEN.wire, f.id,
            ),
        ).thenReturn(true)
        service().dismiss(adminId, f.id, null)
        assertEquals(PhotoStatus.HIDDEN, p.status)
    }

    @Test
    fun `list rejects an unknown status before touching the repository`() {
        assertFailsWith<ValidationException> {
            service().list("bogus", null, PaginationParams.of(0, 10))
        }
        Mockito.verifyNoInteractions(flagRepository)
    }

    @Test
    fun `hydrate falls back when the photo has no derived asset and no owner`() {
        val p = photo(PhotoStatus.PROCESSING)
        val f = flag(FlagStatus.OPEN, p)
        val dto = service().escalate(adminId, f.id, null)
        assertNull(dto.photoSnapshot.thumbnailUrl, "a PROCESSING photo must not presign the original")
        assertEquals("photographer", dto.photographerHandle)
        assertEquals("system", dto.reportedBy)
        assertNull(dto.reviewedBy)
        Mockito.verify(storageService, Mockito.never()).presignedGetUrl(anyString(), anyArg())
    }
}
