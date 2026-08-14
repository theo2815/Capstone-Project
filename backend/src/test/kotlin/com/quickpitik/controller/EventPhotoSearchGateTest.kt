package com.quickpitik.controller

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.service.photos.PhotoSearchService
import com.quickpitik.service.photos.PhotoService
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.mock.web.MockHttpServletRequest
import org.springframework.mock.web.MockMultipartFile
import java.time.LocalDate
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Runner-flow audit (2026-05-27), "Events + photo discovery" item E1.
//
// searchByFaceMultipart forwarded the part straight to ai-api. The only bound
// was Spring's 25 MB multipart ceiling, which is sized for photographer
// originals — so an unauthenticated caller could push 25 MB of arbitrary
// content-type at the inference service. Same whitelist + 5 MB cap the stored
// selfie already cleared in SelfieService.upload.
class EventPhotoSearchGateTest {

    private lateinit var eventRepository: EventRepository
    private lateinit var photoSearchService: PhotoSearchService
    private lateinit var controller: EventPhotoController

    private val slug = "cebu-marathon"

    @BeforeEach
    fun setUp() {
        eventRepository = Mockito.mock(EventRepository::class.java)
        photoSearchService = Mockito.mock(PhotoSearchService::class.java)
        Mockito.`when`(eventRepository.findBySlugAndDeletedAtIsNull(slug)).thenReturn(
            Event(
                slug = slug,
                name = "Cebu Marathon",
                date = LocalDate.of(2026, 1, 11),
                location = "Cebu City, Cebu",
                status = EventStatus.ACTIVE,
            ),
        )
        Mockito.`when`(
            photoSearchService.searchByFace(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg()),
        ).thenReturn(PaginatedResponse(emptyList<PhotoDto>(), 0L, 0, 20))

        controller = EventPhotoController(
            eventRepository,
            Mockito.mock(PhotoService::class.java),
            photoSearchService,
            Mockito.mock(UserSelfieRepository::class.java),
            Mockito.mock(StorageService::class.java),
            allowAll,
        )
    }

    // Rate limiting is a separate concern from the content gate; let everything
    // through so these tests only observe the MIME + size checks.
    private val allowAll = object : RateLimiter {
        override fun tryAcquire(policy: String, key: String) = RateLimiter.Decision(allowed = true)
    }

    @Test
    fun `a non-image part is rejected before ai-api is reached`() {
        val ex = assertFailsWith<ValidationException> {
            search(MockMultipartFile("selfie", "notes.txt", "text/plain", byteArrayOf(1, 2, 3)))
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, ex.code)
        assertEquals("selfie", ex.field)
        verifyNoSearch()
    }

    @Test
    fun `a missing content type is rejected rather than defaulted`() {
        val ex = assertFailsWith<ValidationException> {
            search(MockMultipartFile("selfie", "selfie.jpg", null, byteArrayOf(1, 2, 3)))
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, ex.code)
        verifyNoSearch()
    }

    @Test
    fun `an oversized selfie is rejected with 413`() {
        val ex = assertFailsWith<ApiException> {
            search(MockMultipartFile("selfie", "big.jpg", "image/jpeg", ByteArray(5 * 1024 * 1024 + 1)))
        }

        assertEquals(ErrorCodes.PAYLOAD_TOO_LARGE, ex.code)
        verifyNoSearch()
    }

    @Test
    fun `an empty part is rejected`() {
        val ex = assertFailsWith<ValidationException> {
            search(MockMultipartFile("selfie", "selfie.jpg", "image/jpeg", ByteArray(0)))
        }

        assertEquals(ErrorCodes.VALIDATION_ERROR, ex.code)
        verifyNoSearch()
    }

    @Test
    fun `all three accepted image types pass through`() {
        listOf("image/jpeg", "image/png", "image/webp").forEach { mime ->
            search(MockMultipartFile("selfie", "selfie", mime, byteArrayOf(1, 2, 3)))
        }

        Mockito.verify(photoSearchService, Mockito.times(3))
            .searchByFace(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `a charset parameter on the content type does not break the whitelist`() {
        // Browsers and HTTP clients are free to append parameters.
        search(MockMultipartFile("selfie", "selfie.jpg", "image/jpeg; charset=binary", byteArrayOf(1, 2, 3)))

        Mockito.verify(photoSearchService)
            .searchByFace(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
    }

    private fun search(file: MockMultipartFile) = controller.searchByFaceMultipart(
        principal = null,
        slug = slug,
        selfie = file,
        offset = null,
        limit = null,
        request = MockHttpServletRequest(),
    )

    private fun verifyNoSearch() = Mockito.verify(photoSearchService, Mockito.never())
        .searchByFace(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())

    private fun <T> anyArg(): T = Mockito.any()
}
