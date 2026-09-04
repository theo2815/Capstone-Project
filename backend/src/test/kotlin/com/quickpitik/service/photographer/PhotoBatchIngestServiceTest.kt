package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.AiProperties
import com.quickpitik.dto.ai.JobStatusResult
import com.quickpitik.entity.AiIndexBatch
import com.quickpitik.entity.AiIndexJob
import com.quickpitik.entity.BatchStatus
import com.quickpitik.entity.IndexJobKind
import com.quickpitik.entity.IndexJobStatus
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.repository.AiIndexBatchRepository
import com.quickpitik.repository.AiIndexJobRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.websocket.PhotoIndexedEvent
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import java.math.BigDecimal
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// Phase C ingest — the §3.11 plan-mandated coverage: idempotency, ref→photo
// mapping, two-job finalize matrix, revert-on-failure, and the #2 fix that
// terminates a lost (NOT_FOUND) job instead of polling it forever.
class PhotoBatchIngestServiceTest {

    private lateinit var photoRepo: PhotoRepository
    private lateinit var batchRepo: AiIndexBatchRepository
    private lateinit var jobRepo: AiIndexJobRepository
    private lateinit var aiClient: AiApiClient
    private lateinit var publisher: ApplicationEventPublisher
    private lateinit var service: PhotoBatchIngestService

    private val eventId = UUID.randomUUID()
    private val pid1 = UUID.randomUUID()
    private val pid2 = UUID.randomUUID()
    private lateinit var batch: AiIndexBatch
    private lateinit var photo1: Photo
    private lateinit var photo2: Photo

    @BeforeEach
    fun setUp() {
        photoRepo = Mockito.mock(PhotoRepository::class.java)
        batchRepo = Mockito.mock(AiIndexBatchRepository::class.java)
        jobRepo = Mockito.mock(AiIndexJobRepository::class.java)
        aiClient = Mockito.mock(AiApiClient::class.java)
        publisher = Mockito.mock(ApplicationEventPublisher::class.java)
        service = PhotoBatchIngestService(photoRepo, batchRepo, jobRepo, aiClient, AiApiProperties(), AiProperties(), publisher)

        batch = AiIndexBatch(eventId = eventId, photoIds = mutableListOf(pid1, pid2))
        photo1 = batchingPhoto(pid1)
        photo2 = batchingPhoto(pid2)
    }

    private fun batchingPhoto(id: UUID) =
        Photo(id = id, eventId = eventId, s3Key = "k", pricePhp = BigDecimal.TEN).also {
            it.indexingStatus = IndexingStatus.BATCHING
            it.indexingAttempts = 1
        }

    private fun job(kind: IndexJobKind, aiJobId: String, status: IndexJobStatus = IndexJobStatus.SUBMITTED) =
        AiIndexJob(batchId = batch.id, kind = kind, aiJobId = aiJobId).also { it.status = status }

    private fun expectLookup(faceJob: AiIndexJob) {
        Mockito.`when`(jobRepo.findByAiJobId(faceJob.aiJobId)).thenReturn(faceJob)
        Mockito.`when`(batchRepo.findByIdForUpdate(batch.id)).thenReturn(batch)
        Mockito.`when`(photoRepo.findAllById(batch.photoIds)).thenReturn(mutableListOf(photo1, photo2))
    }

    @Test
    fun `already-terminal job is a no-op`() {
        val faceJob = job(IndexJobKind.FACE, "face-job", IndexJobStatus.COMPLETED)
        Mockito.`when`(jobRepo.findByAiJobId("face-job")).thenReturn(faceJob)

        service.ingest("face-job")

        Mockito.verifyNoInteractions(batchRepo)
        Mockito.verify(aiClient, Mockito.never()).jobStatus("face-job", 0, 2)
    }

    @Test
    fun `completed job writes embeds by ref but does not finalize while sibling is open`() {
        val faceJob = job(IndexJobKind.FACE, "face-job")
        val bibJob = job(IndexJobKind.BIB, "bib-job") // still SUBMITTED
        expectLookup(faceJob)
        Mockito.`when`(aiClient.jobStatus("face-job", 0, 2)).thenReturn(
            JobStatusResult(
                job_id = "face-job",
                status = "completed",
                result = listOf(
                    mapOf("ref" to pid1.toString(), "person_id" to "person-1"),
                    mapOf("ref" to pid2.toString(), "person_id" to "person-2"),
                ),
            ),
        )
        Mockito.`when`(jobRepo.findByBatchId(batch.id)).thenReturn(listOf(faceJob, bibJob))

        service.ingest("face-job")

        assertEquals(IndexJobStatus.COMPLETED, faceJob.status)
        assertEquals(setOf("person-1"), photo1.facePersons.map { it.aiPersonId }.toSet())
        assertEquals(setOf("person-2"), photo2.facePersons.map { it.aiPersonId }.toSet())
        assertEquals(IndexingStatus.BATCHING, photo1.indexingStatus) // not finalized yet
        Mockito.verify(publisher, Mockito.never()).publishEvent(Mockito.any(PhotoIndexedEvent::class.java))
    }

    @Test
    fun `both jobs completed - finalize to INDEXED and notify per photo`() {
        val faceJob = job(IndexJobKind.FACE, "face-job")
        val bibJob = job(IndexJobKind.BIB, "bib-job", IndexJobStatus.COMPLETED)
        expectLookup(faceJob)
        Mockito.`when`(aiClient.jobStatus("face-job", 0, 2)).thenReturn(
            JobStatusResult(
                job_id = "face-job",
                status = "completed",
                result = listOf(mapOf("ref" to pid1.toString(), "person_id" to "person-1")),
            ),
        )
        Mockito.`when`(jobRepo.findByBatchId(batch.id)).thenReturn(listOf(faceJob, bibJob))

        service.ingest("face-job")

        assertEquals(IndexingStatus.INDEXED, photo1.indexingStatus)
        assertEquals(IndexingStatus.INDEXED, photo2.indexingStatus)
        assertEquals(BatchStatus.FINALIZED, batch.status)
        Mockito.verify(publisher, Mockito.times(2)).publishEvent(Mockito.any(PhotoIndexedEvent::class.java))
    }

    @Test
    fun `lost job (NOT_FOUND) is terminated, not polled forever`() {
        val faceJob = job(IndexJobKind.FACE, "face-job")
        val bibJob = job(IndexJobKind.BIB, "bib-job", IndexJobStatus.FAILED)
        expectLookup(faceJob)
        // ai-api reports a gone/expired job as a NOT_FOUND envelope → AiApiException.
        Mockito.`when`(aiClient.jobStatus("face-job", 0, 2))
            .thenThrow(AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "NOT_FOUND", "job not found"))
        Mockito.`when`(jobRepo.findByBatchId(batch.id)).thenReturn(listOf(faceJob, bibJob))

        service.ingest("face-job")

        assertEquals(IndexJobStatus.FAILED, faceJob.status)
        assertTrue(faceJob.error!!.contains("not found"))
        // both jobs failed, attempts (1) < max (5) → revert to PENDING for re-drain
        assertEquals(IndexingStatus.PENDING, photo1.indexingStatus)
        assertEquals(IndexingStatus.PENDING, photo2.indexingStatus)
        assertEquals(BatchStatus.FAILED, batch.status)
    }

    @Test
    fun `a transient ai-api error leaves the job SUBMITTED for the next poll`() {
        val faceJob = job(IndexJobKind.FACE, "face-job")
        expectLookup(faceJob)
        Mockito.`when`(aiClient.jobStatus("face-job", 0, 2))
            .thenThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "ai-api down"))

        service.ingest("face-job")

        assertEquals(IndexJobStatus.SUBMITTED, faceJob.status)
        Mockito.verify(jobRepo, Mockito.never()).save(Mockito.any())
    }
}
