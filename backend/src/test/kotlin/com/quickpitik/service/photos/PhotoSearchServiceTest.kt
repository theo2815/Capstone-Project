package com.quickpitik.service.photos

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.ai.FaceMatch
import com.quickpitik.dto.ai.FacesSearchResult
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.entity.UserSelfie
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.service.storage.StorageService
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

    @Test fun `raw candidates keep configured threshold scores and visible provider limit`() {
        stubSearch(FaceMatch("low", similarity = 0.59), FaceMatch("edge", similarity = 0.6), FaceMatch("high", similarity = 0.94))
        val result = candidates(service())
        assertEquals(listOf("edge", "high"), result["matches"].map { it["personId"].asText() })
        assertEquals(0.94, result["matches"][1]["similarity"].asDouble())
        assertEquals(50, result["candidateLimit"].asInt())
        assertEquals(false, result["possiblyTruncated"].asBoolean())
        Mockito.verifyNoInteractions(photoService, userSelfieRepository, storageService)
    }

    @Test fun `raw candidates report saturation before threshold filtering`() {
        stubSearch(*Array(50) { FaceMatch("p-$it", similarity = if (it == 0) 0.9 else 0.1) })
        val result = candidates(service())
        assertEquals(1, result["matches"].size())
        assertTrue(result["possiblyTruncated"].asBoolean())
    }

    @Test fun `raw candidate errors cannot become a fallback gallery`() {
        whenFacesSearch().thenThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "offline"))
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, assertFailsWith<AiApiException> { candidates(service(fallback = true)) }.status)
        Mockito.verifyNoInteractions(photoService)
    }

    @Test fun `raw candidates reject disabled AI and empty evidence before provider I O`() {
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, assertFailsWith<AiApiException> { candidates(service(enabled = false)) }.status)
        assertFailsWith<com.quickpitik.exception.ValidationException> { candidates(service(), byteArrayOf()) }
        Mockito.verifyNoInteractions(aiApiClient)
    }

    @Test fun `raw candidates preserve provider no face and pre dedup saturation metadata`() {
        val mapper = com.fasterxml.jackson.module.kotlin.jacksonObjectMapper()
        whenFacesSearch().thenReturn(mapper.readValue("{\"matches\":[],\"noFaces\":true,\"possiblyTruncated\":true}", FacesSearchResult::class.java))
        val result = candidates(service())
        assertTrue(result["noFaces"].asBoolean())
        assertTrue(result["possiblyTruncated"].asBoolean())
        // Public hydration retains the existing empty result for this provider outcome.
        assertTrue(service().searchByFace(eventId, selfie, "image/jpeg", "s.jpg", pagination).items.isEmpty())
    }

    private fun candidates(service: PhotoSearchService, bytes: ByteArray = selfie): com.fasterxml.jackson.databind.JsonNode {
        return com.fasterxml.jackson.module.kotlin.jacksonObjectMapper().valueToTree(
            service.findFaceCandidates(eventId, com.quickpitik.dto.removal.SearchImage(bytes, "image/jpeg", "s.jpg")))
    }

    private lateinit var aiApiClient: AiApiClient
    private lateinit var photoService: PhotoService
    private lateinit var userSelfieRepository: UserSelfieRepository
    private lateinit var storageService: StorageService

    private val eventId: UUID = UUID.randomUUID()
    private val userId: UUID = UUID.randomUUID()
    private val pagination = PaginationParams.of(0, 20)
    private val selfie = byteArrayOf(1, 2, 3)

    @BeforeEach
    fun setUp() {
        aiApiClient = Mockito.mock(AiApiClient::class.java)
        photoService = Mockito.mock(PhotoService::class.java)
        userSelfieRepository = Mockito.mock(UserSelfieRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
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
    fun `with the fallback on a rejected selfie still propagates 422`() {
        val service = service(fallback = true)
        whenFacesSearch().thenThrow(AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "NO_FACES", "no faces"))
        Mockito.`when`(photoService.listForEvent(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(PaginatedResponse(emptyList<PhotoDto>(), 7L, 0, 20))

        val ex = assertFailsWith<AiApiException> {
            service.searchByFace(eventId, selfie, "image/jpeg", "s.jpg", pagination)
        }

        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, ex.status)
        assertEquals("NO_FACES", ex.aiCode)
        Mockito.verifyNoInteractions(photoService)
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

    // ─── Stored-selfie entry point (moved out of EventPhotoController 2026-09-12) ─

    @Test
    fun `all-selfies search with an empty library is a 404`() {
        val service = service()
        Mockito.`when`(userSelfieRepository.findByUserIdOrderByUploadedAtDesc(userId)).thenReturn(emptyList())

        val ex = assertFailsWith<NotFoundException> {
            service.searchWithStoredSelfie(userId, eventId, null, allSelfies = true, pagination = pagination)
        }

        assertEquals(ErrorCodes.SELFIE_NOT_FOUND, ex.code)
        // Both SELFIE_NOT_FOUND branches share the code; the message is what
        // separates "no selfies saved" from "the bytes went missing".
        assertEquals("Selfie not found", ex.message)
        assertTrue(Mockito.mockingDetails(aiApiClient).invocations.isEmpty())
    }

    @Test
    fun `all-selfies search whose every object is gone is a 404, not a 500`() {
        val service = service()
        Mockito.`when`(userSelfieRepository.findByUserIdOrderByUploadedAtDesc(userId))
            .thenReturn(listOf(selfieRow("s1.jpg"), selfieRow("s2.jpg")))
        Mockito.`when`(storageService.getBytes(anyArg())).thenThrow(IllegalStateException("object missing"))

        val ex = assertFailsWith<NotFoundException> {
            service.searchWithStoredSelfie(userId, eventId, null, allSelfies = true, pagination = pagination)
        }

        assertEquals(ErrorCodes.SELFIE_NOT_FOUND, ex.code)
        assertEquals("Selfie file is no longer available", ex.message)
        assertTrue(Mockito.mockingDetails(aiApiClient).invocations.isEmpty())
    }

    @Test
    fun `a single stored selfie whose bytes are gone is a 404, not a 500`() {
        val service = service()
        val row = selfieRow("s.jpg")
        Mockito.`when`(userSelfieRepository.findByIdAndUserId(row.id, userId)).thenReturn(row)
        Mockito.`when`(storageService.getBytes(row.s3Key)).thenThrow(IllegalStateException("object missing"))

        val ex = assertFailsWith<NotFoundException> {
            service.searchWithStoredSelfie(userId, eventId, row.id, allSelfies = false, pagination = pagination)
        }

        assertEquals(ErrorCodes.SELFIE_NOT_FOUND, ex.code)
        assertEquals("Selfie file is no longer available", ex.message)
        assertTrue(Mockito.mockingDetails(aiApiClient).invocations.isEmpty())
    }

    @Test
    fun `the all-selfies library is capped at five samples`() {
        val service = service()
        // Seven saved selfies must not become seven AI calls (one per sample).
        Mockito.`when`(userSelfieRepository.findByUserIdOrderByUploadedAtDesc(userId))
            .thenReturn((1..7).map { selfieRow("s$it.jpg") })
        Mockito.`when`(storageService.getBytes(anyArg())).thenReturn(selfie)
        stubSearch()

        service.searchWithStoredSelfie(userId, eventId, null, allSelfies = true, pagination = pagination)

        Mockito.verify(aiApiClient, Mockito.times(5)).facesSearch(
            anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyDouble(), Mockito.anyInt(),
        )
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private fun service(enabled: Boolean = true, fallback: Boolean = false): PhotoSearchService =
        PhotoSearchService(
            aiApiClient,
            AiApiProperties(enabled = enabled, searchFallbackOnError = fallback),
            photoService,
            userSelfieRepository,
            storageService,
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

    private fun selfieRow(key: String = "s.jpg") = UserSelfie(userId = userId, s3Key = "selfies/$userId/$key")

    private fun <T> anyArg(): T = Mockito.any()

    // Mockito's capture() returns null; declared as platform T so Kotlin inserts
    // no null check on the non-null parameter it stands in for.
    private fun <T> capture(c: org.mockito.ArgumentCaptor<T>): T = c.capture()

    // Mockito.eq returns a platform-typed null that Kotlin's non-null check
    // rejects before Mockito can register the matcher. Falling back to `value`
    // keeps the type non-null; the matcher is already recorded either way.
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
