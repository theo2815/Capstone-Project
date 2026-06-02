package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.ai.JobCreateResult
import com.quickpitik.entity.AiIndexBatch
import com.quickpitik.entity.AiIndexJob
import com.quickpitik.entity.IndexJobKind
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.repository.AiIndexBatchRepository
import com.quickpitik.repository.AiIndexJobRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.data.domain.PageRequest
import org.springframework.http.HttpStatus
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// Phase C batch drain — the §3.11 plan-mandated coverage: group one event's
// backlog into a batch + 2 jobs (ordered photoIds, flip BATCHING, attempts++),
// the single-event mega stamping (cross-event isolation gate), and the
// all-in-one-tx rollback semantics when a mega POST fails.
class PhotoBatchIndexingServiceTest {

    private lateinit var photoRepo: PhotoRepository
    private lateinit var batchRepo: AiIndexBatchRepository
    private lateinit var jobRepo: AiIndexJobRepository
    private lateinit var storage: StorageService
    private lateinit var aiClient: AiApiClient
    private lateinit var service: PhotoBatchIndexingService

    private val eventId = UUID.randomUUID()
    private val cutoff = OffsetDateTime.now()
    private val jpeg1 = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0x01)
    private val jpeg2 = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0x02)

    // Mockito.any() returns null at runtime; declared as a (platform) T so Kotlin
    // inserts no null-assertion — usable as a matcher for non-null params.
    private fun <T> anyArg(): T = Mockito.any()

    // Same trick for ArgumentCaptor.capture(): the declared T return lets Kotlin
    // pass the (null) result to a non-null Kotlin parameter without a checkNotNull.
    private fun <T> capture(c: ArgumentCaptor<T>): T = c.capture()

    @BeforeEach
    fun setUp() {
        photoRepo = Mockito.mock(PhotoRepository::class.java)
        batchRepo = Mockito.mock(AiIndexBatchRepository::class.java)
        jobRepo = Mockito.mock(AiIndexJobRepository::class.java)
        storage = Mockito.mock(StorageService::class.java)
        aiClient = Mockito.mock(AiApiClient::class.java)
        service = PhotoBatchIndexingService(photoRepo, batchRepo, jobRepo, storage, aiClient, AiApiProperties())
    }

    private fun pendingPhoto(key: String) =
        Photo(eventId = eventId, s3Key = key, pricePhp = BigDecimal.TEN) // PENDING, attempts 0 by default

    private fun stubBacklog(vararg photos: Photo) {
        Mockito.`when`(
            photoRepo.findIndexingBacklogForEvent(
                eventId,
                listOf(IndexingStatus.PENDING, IndexingStatus.FAILED),
                5,
                cutoff,
                PageRequest.of(0, 50),
            ),
        ).thenReturn(photos.toList())
    }

    @Test
    fun `drain creates a batch plus FACE and BIB jobs and flips photos to BATCHING`() {
        val p1 = pendingPhoto("k1")
        val p2 = pendingPhoto("k2")
        stubBacklog(p1, p2)
        Mockito.`when`(storage.getBytes("k1")).thenReturn(jpeg1)
        Mockito.`when`(storage.getBytes("k2")).thenReturn(jpeg2)
        Mockito.`when`(aiClient.bibsRecognizeMega(anyArg())).thenReturn(JobCreateResult("bib-1"))
        Mockito.`when`(aiClient.facesEnrollMega(anyArg(), anyArg())).thenReturn(JobCreateResult("face-1"))

        val drained = service.drainEvent(eventId, cutoff)

        assertTrue(drained)
        assertEquals(IndexingStatus.BATCHING, p1.indexingStatus)
        assertEquals(IndexingStatus.BATCHING, p2.indexingStatus)
        assertEquals(1, p1.indexingAttempts)
        assertEquals(1, p2.indexingAttempts)

        val batchCaptor = ArgumentCaptor.forClass(AiIndexBatch::class.java)
        Mockito.verify(batchRepo).save(capture(batchCaptor))
        assertEquals(eventId, batchCaptor.value.eventId)
        assertEquals(listOf(p1.id, p2.id), batchCaptor.value.photoIds) // ordered

        val jobCaptor = ArgumentCaptor.forClass(AiIndexJob::class.java)
        Mockito.verify(jobRepo, Mockito.times(2)).save(capture(jobCaptor))
        val saved = jobCaptor.allValues
        assertEquals("face-1", saved.first { it.kind == IndexJobKind.FACE }.aiJobId)
        assertEquals("bib-1", saved.first { it.kind == IndexJobKind.BIB }.aiJobId)
    }

    @Test
    fun `the face mega is stamped with the batch's single event id - cross-event isolation gate`() {
        val p1 = pendingPhoto("k1")
        stubBacklog(p1)
        Mockito.`when`(storage.getBytes("k1")).thenReturn(jpeg1)
        Mockito.`when`(aiClient.bibsRecognizeMega(anyArg())).thenReturn(JobCreateResult("bib-1"))
        Mockito.`when`(aiClient.facesEnrollMega(anyArg(), anyArg())).thenReturn(JobCreateResult("face-1"))

        service.drainEvent(eventId, cutoff)

        // The drain groups strictly by event, and the FACE mega carries that one
        // event_id — so a runner in Event A vs B becomes two event-scoped persons.
        val evCaptor: ArgumentCaptor<UUID> = ArgumentCaptor.forClass(UUID::class.java)
        Mockito.verify(aiClient).facesEnrollMega(anyArg(), capture(evCaptor))
        assertEquals(eventId, evCaptor.value)
    }

    @Test
    fun `a mega POST failure leaves photos PENDING and creates no batch (rollback semantics)`() {
        val p1 = pendingPhoto("k1")
        stubBacklog(p1)
        Mockito.`when`(storage.getBytes("k1")).thenReturn(jpeg1)
        // bib mega is submitted first; if it throws, nothing downstream runs.
        Mockito.`when`(aiClient.bibsRecognizeMega(anyArg()))
            .thenThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "ai-api down"))

        assertThrows<AiApiException> { service.drainEvent(eventId, cutoff) }

        assertEquals(IndexingStatus.PENDING, p1.indexingStatus) // not flipped
        assertEquals(0, p1.indexingAttempts) // attempt not consumed
        Mockito.verify(batchRepo, Mockito.never()).save(Mockito.any())
        Mockito.verify(jobRepo, Mockito.never()).save(Mockito.any())
    }

    @Test
    fun `empty backlog drains nothing and calls no ai-api`() {
        stubBacklog()

        val drained = service.drainEvent(eventId, cutoff)

        assertFalse(drained)
        Mockito.verifyNoInteractions(aiClient)
        Mockito.verify(batchRepo, Mockito.never()).save(Mockito.any())
    }
}
