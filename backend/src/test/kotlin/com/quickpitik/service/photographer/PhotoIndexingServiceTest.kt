package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.AiProperties
import com.quickpitik.dto.ai.BibDetection
import com.quickpitik.dto.ai.BibsRecognizeResult
import com.quickpitik.dto.ai.FacesEnrollResult
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoFacePersonEmbed
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.storage.StorageService
import com.quickpitik.websocket.PhotoIndexedEvent
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

// Phase B per-photo indexer — the plan-mandated unit coverage (enterprise-plan
// §"Tests (backend)"): one enroll + one bib per photo, benign NO_FACES/LOW_QUALITY
// is not a failure, the INDEXED/PARTIAL/FAILED matrix, idempotent re-run, and the
// event-isolation enroll stamping. Also guards the #6 bib-confidence fix.
class PhotoIndexingServiceTest {

    private lateinit var photoRepo: PhotoRepository
    private lateinit var storage: StorageService
    private lateinit var aiClient: AiApiClient
    private lateinit var publisher: ApplicationEventPublisher
    private lateinit var service: PhotoIndexingService

    // FF D8 FF → sniffed as image/jpeg by PhotoIndexingService.sniffContentType.
    private val jpegBytes = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0x00)

    private fun props(enabled: Boolean = true) = AiApiProperties(enabled = enabled)

    // Real template over a stubbed manager: execute { } runs the callback inline
    // (and rethrows), so the unit tests stay transaction-free. getTransaction
    // must return a real status — a raw mock's null trips the Kotlin lambda's
    // non-null parameter check.
    private fun template() = TransactionTemplate(
        Mockito.mock(PlatformTransactionManager::class.java).also {
            Mockito.`when`(it.getTransaction(Mockito.any())).thenReturn(SimpleTransactionStatus())
        },
    )

    @BeforeEach
    fun setUp() {
        photoRepo = Mockito.mock(PhotoRepository::class.java)
        storage = Mockito.mock(StorageService::class.java)
        aiClient = Mockito.mock(AiApiClient::class.java)
        publisher = Mockito.mock(ApplicationEventPublisher::class.java)
        service = PhotoIndexingService(photoRepo, storage, aiClient, props(), AiProperties(), publisher, template())
    }

    private fun photo(status: IndexingStatus = IndexingStatus.PENDING): Photo =
        Photo(eventId = UUID.randomUUID(), s3Key = "orig/x.jpg", pricePhp = BigDecimal.TEN).also {
            it.indexingStatus = status
        }

    private fun expectPhoto(p: Photo) {
        Mockito.`when`(photoRepo.findById(p.id)).thenReturn(Optional.of(p))
        Mockito.`when`(storage.getBytes(p.s3Key)).thenReturn(jpegBytes)
    }

    private fun stubEnrollReturns(p: Photo, personId: String) {
        Mockito.`when`(
            aiClient.facesEnroll(jpegBytes, "image/jpeg", "${p.id}.jpg", p.id.toString(), null, p.eventId),
        ).thenReturn(FacesEnrollResult(person_id = personId))
    }

    private fun stubBibsReturns(p: Photo, result: BibsRecognizeResult) {
        Mockito.`when`(aiClient.bibsRecognize(jpegBytes, "image/jpeg", "${p.id}.jpg")).thenReturn(result)
    }

    @Test
    fun `both calls succeed - INDEXED with one face person and the bib`() {
        val p = photo()
        expectPhoto(p)
        stubEnrollReturns(p, "person-1")
        stubBibsReturns(p, BibsRecognizeResult(listOf(BibDetection("1234", 0.92))))

        service.index(p.id)

        assertEquals(IndexingStatus.INDEXED, p.indexingStatus)
        assertEquals(1, p.indexingAttempts)
        assertEquals(setOf("person-1"), p.facePersons.map { it.aiPersonId }.toSet())
        assertEquals(setOf("1234"), p.bibs.map { it.bibNumber }.toSet())
        assertTrue(p.indexedAt != null)
        // V33 provider stamp — which ID space the stored person id belongs to.
        assertEquals("ai_api", p.indexedProvider)
        Mockito.verify(publisher).publishEvent(Mockito.any(PhotoIndexedEvent::class.java))
    }

    @Test
    fun `NO_FACES is benign - still INDEXED, no face person`() {
        val p = photo()
        expectPhoto(p)
        Mockito.`when`(
            aiClient.facesEnroll(jpegBytes, "image/jpeg", "${p.id}.jpg", p.id.toString(), null, p.eventId),
        ).thenThrow(AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "NO_FACES", "no faces detected"))
        stubBibsReturns(p, BibsRecognizeResult(emptyList()))

        service.index(p.id)

        assertEquals(IndexingStatus.INDEXED, p.indexingStatus)
        assertTrue(p.facePersons.isEmpty())
    }

    @Test
    fun `LOW_QUALITY is benign - not FAILED`() {
        val p = photo()
        expectPhoto(p)
        Mockito.`when`(
            aiClient.facesEnroll(jpegBytes, "image/jpeg", "${p.id}.jpg", p.id.toString(), null, p.eventId),
        ).thenThrow(AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "LOW_QUALITY", "faces too small"))
        stubBibsReturns(p, BibsRecognizeResult(listOf(BibDetection("77", 0.9))))

        service.index(p.id)

        assertEquals(IndexingStatus.INDEXED, p.indexingStatus)
    }

    @Test
    fun `one half hits a transient error - PARTIAL`() {
        val p = photo()
        expectPhoto(p)
        Mockito.`when`(
            aiClient.facesEnroll(jpegBytes, "image/jpeg", "${p.id}.jpg", p.id.toString(), null, p.eventId),
        ).thenThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "ai-api down"))
        stubBibsReturns(p, BibsRecognizeResult(listOf(BibDetection("9", 0.9))))

        service.index(p.id)

        assertEquals(IndexingStatus.PARTIAL, p.indexingStatus)
        assertNull(p.indexedAt)
        assertTrue(p.indexingError!!.contains("faces"))
    }

    @Test
    fun `both halves transport-fail - back to PENDING, attempt not consumed`() {
        // Provider fully unreachable (both calls die retryably): the photo must
        // NOT burn an attempt — a ~5-min outage used to exhaust the 5-attempt
        // budget and strand photos terminally FAILED (2026-08-25 incident).
        val p = photo()
        expectPhoto(p)
        Mockito.`when`(
            aiClient.facesEnroll(jpegBytes, "image/jpeg", "${p.id}.jpg", p.id.toString(), null, p.eventId),
        ).thenThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "down"))
        Mockito.`when`(aiClient.bibsRecognize(jpegBytes, "image/jpeg", "${p.id}.jpg"))
            .thenThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "down"))

        service.index(p.id)

        assertEquals(IndexingStatus.PENDING, p.indexingStatus)
        assertEquals(0, p.indexingAttempts)
        Mockito.verify(publisher, Mockito.never()).publishEvent(Mockito.any(PhotoIndexedEvent::class.java))
    }

    @Test
    fun `both halves fail non-retryably - FAILED, attempt consumed, no notification`() {
        val p = photo()
        expectPhoto(p)
        Mockito.`when`(
            aiClient.facesEnroll(jpegBytes, "image/jpeg", "${p.id}.jpg", p.id.toString(), null, p.eventId),
        ).thenThrow(AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, null, "bad image"))
        Mockito.`when`(aiClient.bibsRecognize(jpegBytes, "image/jpeg", "${p.id}.jpg"))
            .thenThrow(AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, null, "bad image"))

        service.index(p.id)

        assertEquals(IndexingStatus.FAILED, p.indexingStatus)
        assertEquals(1, p.indexingAttempts)
        Mockito.verify(publisher, Mockito.never()).publishEvent(Mockito.any(PhotoIndexedEvent::class.java))
    }

    @Test
    fun `missing original bytes - FAILED before any ai call`() {
        val p = photo()
        Mockito.`when`(photoRepo.findById(p.id)).thenReturn(Optional.of(p))
        Mockito.`when`(storage.getBytes(p.s3Key)).thenThrow(RuntimeException("gone"))

        service.index(p.id)

        assertEquals(IndexingStatus.FAILED, p.indexingStatus)
        assertTrue(p.indexingError!!.startsWith("original image unavailable"))
        Mockito.verifyNoInteractions(aiClient)
    }

    @Test
    fun `ai-api disabled - SKIPPED, no calls`() {
        val p = photo()
        Mockito.`when`(photoRepo.findById(p.id)).thenReturn(Optional.of(p))
        val disabled =
            PhotoIndexingService(photoRepo, storage, aiClient, props(enabled = false), AiProperties(), publisher, template())

        disabled.index(p.id)

        assertEquals(IndexingStatus.SKIPPED, p.indexingStatus)
        Mockito.verifyNoInteractions(aiClient)
        Mockito.verifyNoInteractions(storage)
    }

    @Test
    fun `re-run is idempotent - prior ai person erased, no duplicate`() {
        val p = photo(status = IndexingStatus.FAILED).also {
            it.facePersons.add(PhotoFacePersonEmbed(faceIndex = 0, aiPersonId = "old-person"))
        }
        expectPhoto(p)
        stubEnrollReturns(p, "new-person")
        stubBibsReturns(p, BibsRecognizeResult(emptyList()))

        service.index(p.id)

        Mockito.verify(aiClient).deleteFacesPerson("old-person")
        assertEquals(setOf("new-person"), p.facePersons.map { it.aiPersonId }.toSet())
    }

    @Test
    fun `enroll is stamped with the photo's own event id - cross-event isolation gate`() {
        val p = photo()
        expectPhoto(p)
        stubEnrollReturns(p, "person-1")
        stubBibsReturns(p, BibsRecognizeResult(emptyList()))

        service.index(p.id)

        // Gate 1: the photo enrolls under its OWN event_id, so the same runner in
        // two events becomes two event-scoped persons — never one shared identity.
        // (The verify matches only when eventId == p.eventId; a different event_id
        // would leave facesEnroll unstubbed and fail the run.)
        Mockito.verify(aiClient)
            .facesEnroll(jpegBytes, "image/jpeg", "${p.id}.jpg", p.id.toString(), null, p.eventId)
    }

    @Test
    fun `duplicate bib stores the max qualifying confidence, not a below-threshold one`() {
        val p = photo()
        expectPhoto(p)
        stubEnrollReturns(p, "person-1")
        // Same bib twice: 0.50 (below the 0.70 default) and 0.90 (qualifies). The
        // stored confidence must be the qualifying 0.90 — never the 0.50 that the
        // old first()-over-unfiltered-list lookup would have persisted.
        stubBibsReturns(p, BibsRecognizeResult(listOf(BibDetection("42", 0.50), BibDetection("42", 0.90))))

        service.index(p.id)

        assertEquals(1, p.bibs.size)
        val bib = p.bibs.first()
        assertEquals("42", bib.bibNumber)
        assertEquals(0, bib.ocrConfidence.compareTo(BigDecimal.valueOf(0.90)))
    }

    @Test
    fun `below-threshold bib is not stored`() {
        val p = photo()
        expectPhoto(p)
        stubEnrollReturns(p, "person-1")
        stubBibsReturns(p, BibsRecognizeResult(listOf(BibDetection("99", 0.50))))

        service.index(p.id)

        assertTrue(p.bibs.isEmpty())
    }
}
