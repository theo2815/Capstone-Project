package com.quickpitik.service.photos

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.ai.FaceMatch
import com.quickpitik.dto.ai.FacesSearchResult
import com.quickpitik.dto.photos.PhotoDto
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiApiException
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// Runner-flow audit (2026-05-27), "Events + photo discovery" items E2 + E4.
//
// E2 — the backend told ai-api which threshold to apply and then trusted the
// answer. A sub-threshold match surfacing means one runner sees another
// runner's photos, which is the single outcome face search must never produce.
//
// E4 — every ai-api failure returned the event's WHOLE grid, commented as a
// demo safety net. That is now behind `search-fallback-on-error`, default off,
// and the real exception (with its real status) propagates instead.
class PhotoSearchServiceTest {

    private lateinit var aiApiClient: AiApiClient
    private lateinit var photoService: PhotoService

    private val eventId: UUID = UUID.randomUUID()
    private val pagination = PaginationParams.of(0, 20)
    private val selfie = byteArrayOf(1, 2, 3)

    @BeforeEach
    fun setUp() {
        aiApiClient = Mockito.mock(AiApiClient::class.java)
        photoService = Mockito.mock(PhotoService::class.java)
    }

    // ─── E2: threshold is re-checked server-side ──────────────────────────

    @Test
    fun `sub-threshold matches are dropped before the photo lookup`() {
        val service = service()
        stubSearch(FaceMatch(person_id = "p-weak", similarity = 0.41))

        val result = service.searchByFace(eventId, selfie, "image/jpeg", "s.jpg", pagination)

        // Everything ai-api returned was below 0.6, so nothing qualifies and we
        // never reach findByEventAndPersonIds.
        assertEquals(emptyList(), result.items)
        assertEquals(0L, result.total)
        Mockito.verify(photoService, Mockito.never())
            .findByEventAndPersonIds(anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `a match exactly at the threshold is kept`() {
        val service = service()
        stubSearch(FaceMatch(person_id = "p-edge", similarity = 0.6))
        stubPhotoLookup()

        service.searchByFace(eventId, selfie, "image/jpeg", "s.jpg", pagination)

        Mockito.verify(photoService).findByEventAndPersonIds(
            eqArg(eventId),
            eqArg(setOf("p-edge")),
            anyArg(),
            anyArg(),
        )
    }

    @Test
    fun `a mixed response keeps only the qualifying person ids`() {
        val service = service()
        stubSearch(
            FaceMatch(person_id = "p-good", similarity = 0.92),
            FaceMatch(person_id = "p-weak", similarity = 0.30),
        )
        stubPhotoLookup()

        service.searchByFace(eventId, selfie, "image/jpeg", "s.jpg", pagination)

        Mockito.verify(photoService).findByEventAndPersonIds(
            eqArg(eventId),
            eqArg(setOf("p-good")),
            anyArg(),
            anyArg(),
        )
    }

    // ─── E4: the fallback is opt-in, and status survives when it is off ───

    @Test
    fun `with the fallback off an ai-api failure propagates its own status`() {
        val service = service(fallback = false)
        whenFacesSearch().thenThrow(AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "NO_FACES", "no faces"))

        val ex = assertFailsWith<AiApiException> {
            service.searchByFace(eventId, selfie, "image/jpeg", "s.jpg", pagination)
        }

        // Rethrown untouched so GlobalExceptionHandler can render it as a 422
        // ("bad selfie") rather than a 503 ("service down, retry").
        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, ex.status)
        assertEquals("NO_FACES", ex.aiCode)
        Mockito.verify(photoService, Mockito.never()).listForEvent(anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `with the fallback off a transport failure becomes a 503`() {
        val service = service(fallback = false)
        whenFacesSearch().thenThrow(IllegalStateException("connection reset"))

        val ex = assertFailsWith<AiApiException> {
            service.searchByFace(eventId, selfie, "image/jpeg", "s.jpg", pagination)
        }

        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, ex.status)
        assertEquals(null, ex.aiCode)
    }

    @Test
    fun `with the fallback on an ai-api failure returns the full event grid`() {
        val service = service(fallback = true)
        whenFacesSearch().thenThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "offline"))
        Mockito.`when`(photoService.listForEvent(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(PaginatedResponse(emptyList<PhotoDto>(), 7L, 0, 20))

        val result = service.searchByFace(eventId, selfie, "image/jpeg", "s.jpg", pagination)

        assertEquals(7L, result.total)
        Mockito.verify(photoService).listForEvent(eqArg(eventId), Mockito.isNull(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `a strict search never uses the configured full-grid fallback`() {
        val service = service(fallback = true)
        whenFacesSearch().thenThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "offline"))

        assertFailsWith<AiApiException> {
            service.searchByFace(
                eventId, selfie, "image/jpeg", "s.jpg", pagination,
                allowFallbackOnError = false,
            )
        }

        Mockito.verify(photoService, Mockito.never()).listForEvent(anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `ai-api disabled short-circuits to 503 before any call is made`() {
        val service = service(enabled = false)

        val ex = assertFailsWith<AiApiException> {
            service.searchByFace(eventId, selfie, "image/jpeg", "s.jpg", pagination)
        }

        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, ex.status)
        assertTrue(Mockito.mockingDetails(aiApiClient).invocations.isEmpty())
    }

    // ─── Whole-library search (2026-09-02) ────────────────────────────────

    @Test
    fun `searching with all selfies unions the person ids and skips a rejected selfie`() {
        val service = service()
        // Selfie 1 → person A; selfie 2 rejected by the provider (no face, 4xx);
        // selfie 3 → persons A + B. Expect exactly {A, B}, one photo lookup.
        whenFacesSearch()
            .thenReturn(FacesSearchResult(matches = listOf(FaceMatch(person_id = "p-a", similarity = 0.9))))
            .thenThrow(AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "NO_FACES", "no face"))
            .thenReturn(
                FacesSearchResult(
                    matches = listOf(
                        FaceMatch(person_id = "p-a", similarity = 0.8),
                        FaceMatch(person_id = "p-b", similarity = 0.7),
                        FaceMatch(person_id = "p-weak", similarity = 0.2),
                    ),
                ),
            )
        stubPhotoLookup()
        val samples = (1..3).map { PhotoSearchService.SelfieSample(byteArrayOf(it.toByte()), "image/jpeg", "s$it.jpg") }

        service.searchByFaces(eventId, samples, pagination)

        @Suppress("UNCHECKED_CAST")
        val captor = org.mockito.ArgumentCaptor.forClass(Collection::class.java)
            as org.mockito.ArgumentCaptor<Collection<String>>
        Mockito.verify(photoService).findByEventAndPersonIds(anyArg(), capture(captor), anyArg(), anyArg())
        assertEquals(setOf("p-a", "p-b"), captor.value.toSet())
    }

    @Test
    fun `all-selfies search fails only when every selfie was rejected`() {
        val service = service()
        whenFacesSearch().thenThrow(AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "NO_FACES", "no face"))
        val samples = listOf(PhotoSearchService.SelfieSample(selfie, "image/jpeg", "s.jpg"))

        val ex = org.junit.jupiter.api.assertThrows<AiApiException> {
            service.searchByFaces(eventId, samples, pagination)
        }
        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, ex.status)
        Mockito.verify(photoService, Mockito.never())
            .findByEventAndPersonIds(anyArg(), anyArg(), anyArg(), anyArg())
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private fun service(enabled: Boolean = true, fallback: Boolean = false): PhotoSearchService =
        PhotoSearchService(
            aiApiClient,
            AiApiProperties(enabled = enabled, searchFallbackOnError = fallback),
            photoService,
            SimpleMeterRegistry(),
        )

    /**
     * `facesSearch` takes a primitive Double threshold and Int topK, so those
     * two positions need the primitive matchers — `any()` hands back null and
     * NPEs on unboxing.
     */
    private fun whenFacesSearch(): org.mockito.stubbing.OngoingStubbing<FacesSearchResult> =
        Mockito.`when`(
            aiApiClient.facesSearch(
                anyArg(), anyArg(), anyArg(), anyArg(),
                Mockito.anyDouble(), Mockito.anyInt(),
            ),
        )

    private fun stubSearch(vararg matches: FaceMatch) {
        whenFacesSearch().thenReturn(FacesSearchResult(matches = matches.toList()))
    }

    private fun stubPhotoLookup() {
        Mockito.`when`(photoService.findByEventAndPersonIds(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(PaginatedResponse(emptyList<PhotoDto>(), 1L, 0, 20))
    }

    private fun <T> anyArg(): T = Mockito.any()

    // Mockito's capture() returns null; declared as platform T so Kotlin inserts
    // no null check on the non-null parameter it stands in for.
    private fun <T> capture(c: org.mockito.ArgumentCaptor<T>): T = c.capture()

    // Mockito.eq returns a platform-typed null that Kotlin's non-null check
    // rejects before Mockito can register the matcher. Falling back to `value`
    // keeps the type non-null; the matcher is already recorded either way.
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
