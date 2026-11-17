package com.quickpitik.service.photos

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.config.*
import com.quickpitik.entity.Photo
import com.quickpitik.entity.RemovalState
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoWorkSnapshot
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import java.math.BigDecimal
import java.time.*
import java.util.UUID
import kotlin.test.*

class PhotoWorkFenceTest {
    private val photos = mock(PhotoRepository::class.java)
    private val now = Instant.parse("2026-10-06T00:00:00Z")
    private val photo = Photo(eventId = UUID.randomUUID(), s3Key = "synthetic/original", pricePhp = BigDecimal.ONE)
    private fun <T> anyArg(): T = any()
    private fun fence(provider: AiProvider = AiProvider.REKOGNITION, mode: IndexingMode = IndexingMode.PER_PHOTO) = PhotoWorkFence(photos,
        StorageProperties(operationTimeout = Duration.ofSeconds(7)), AiApiProperties(indexingMode = mode), AiProperties(provider = provider),
        RekognitionProperties(operationTimeout = Duration.ofSeconds(11)), ObjectMapper(), Clock.fixed(now, ZoneOffset.UTC))

    @Test fun `claims derive deadlines from configured operation budgets and preserve known targets`() {
        `when`(photos.findAllByIdForUpdate(listOf(photo.id))).thenReturn(listOf(photo))
        `when`(photos.claimWork(anyArg(), anyArg(), anyArg(), anyArg())).thenReturn(1)
        val watermark = assertNotNull(fence().claim(photo.id, PhotoWorkKind.WATERMARK))
        val indexing = assertNotNull(fence().claim(photo.id, PhotoWorkKind.INDEXING))
        assertEquals(now.plusSeconds(21), watermark.expiresAt)
        assertEquals(now.plusSeconds(40), indexing.expiresAt)
        verify(photos).recordWork(watermark.token, photo.id, photo.eventId, 0, "WATERMARK", "storage",
            OffsetDateTime.ofInstant(watermark.expiresAt, ZoneOffset.UTC),
            "events/${photo.eventId}/photos/${photo.id}/watermark.jpg", null, null, "[]")
        verify(photos).recordWork(indexing.token, photo.id, photo.eventId, 0, "INDEXING", "rekognition",
            OffsetDateTime.ofInstant(indexing.expiresAt, ZoneOffset.UTC), null, photo.id.toString(), null, "[]")
    }

    @Test fun `restricted or unresolved work cannot acquire another claim`() {
        `when`(photos.findAllByIdForUpdate(listOf(photo.id))).thenReturn(listOf(photo))
        photo.removalState = RemovalState.REVIEW
        assertNull(fence().claim(photo.id, PhotoWorkKind.WATERMARK))
        photo.removalState = RemovalState.NONE
        `when`(photos.hasUnfinishedWork(photo.id, "WATERMARK")).thenReturn(true)
        assertNull(fence().claim(photo.id, PhotoWorkKind.WATERMARK))
        verify(photos, never()).claimWork(anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test fun `batch renewal cannot revive expired restricted or replaced work`() {
        val token = UUID.randomUUID()
        val claim = PhotoWorkClaim(photo.id, 0, token, now.plusSeconds(10), PhotoWorkKind.INDEXING)
        `when`(photos.findAllByIdForUpdate(listOf(photo.id))).thenReturn(listOf(photo))
        photo.indexingWorkToken = token
        photo.indexingWorkExpiresAt = OffsetDateTime.ofInstant(now.minusSeconds(1), ZoneOffset.UTC)
        fence().renewBatchClaim(claim)
        photo.indexingWorkExpiresAt = OffsetDateTime.ofInstant(claim.expiresAt, ZoneOffset.UTC)
        photo.removalState = RemovalState.REVIEW
        fence().renewBatchClaim(claim)
        photo.removalState = RemovalState.NONE
        photo.workGeneration = 1
        fence().renewBatchClaim(claim)
        photo.workGeneration = 0
        photo.indexingWorkToken = UUID.randomUUID()
        fence().renewBatchClaim(claim)
        verify(photos, never()).renewIndexingWork(anyArg(), anyArg(), anyArg())
        photo.indexingWorkToken = token
        fence().renewBatchClaim(claim)
        verify(photos).renewIndexingWork(photo.id, token, OffsetDateTime.ofInstant(now.plusSeconds(40), ZoneOffset.UTC))
    }

    @Test fun `late callback attribution survives expiry and finish retains its target record`() {
        val token = UUID.randomUUID()
        val claim = PhotoWorkClaim(photo.id, 0, token, now.minusSeconds(1), PhotoWorkKind.INDEXING)
        photo.indexingWorkToken = token
        photo.indexingWorkExpiresAt = OffsetDateTime.ofInstant(claim.expiresAt, ZoneOffset.UTC)
        `when`(photos.findAllByIdForUpdate(listOf(photo.id))).thenReturn(listOf(photo))
        val batchId = UUID.randomUUID()
        val snapshot = mock(PhotoWorkSnapshot::class.java)
        `when`(snapshot.photoId).thenReturn(photo.id)
        `when`(snapshot.token).thenReturn(token)
        `when`(snapshot.kind).thenReturn("INDEXING")
        `when`(snapshot.expiresAt).thenReturn(claim.expiresAt)
        `when`(photos.workForBatch(batchId)).thenReturn(listOf(snapshot))
        val fence = fence()
        assertFalse(fence.isCurrent(claim))
        fence.attributeBatch(claim, batchId)
        fence.recordJob(claim, "FACE", "remote-job")
        fence.recordPerson(claim, "late-person")
        assertEquals(listOf(claim), fence.forBatch(batchId))
        fence.finish(claim)
        verify(photos).attributeWorkBatch(token, batchId)
        verify(photos).recordWorkJob(token, "{\"FACE\":\"remote-job\"}")
        verify(photos).recordWorkPersons(token, "[\"late-person\"]")
        verify(photos).finishWorkOutput(token, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))
        verify(photos).finishWork(photo.id, token)
    }

    // WM-1: a failed watermark PUT keeps its claim; once both PUTs (7 s each) must have ended, the claim is released.
    @Test fun `a watermark claim two storage timeouts past expiry is released and a new claim granted`() {
        val stale = UUID.randomUUID()
        photo.watermarkWorkToken = stale
        photo.watermarkWorkExpiresAt = OffsetDateTime.ofInstant(now.minusSeconds(15), ZoneOffset.UTC)
        `when`(photos.findAllByIdForUpdate(listOf(photo.id))).thenReturn(listOf(photo))
        `when`(photos.finishWork(photo.id, stale)).thenAnswer {
            photo.watermarkWorkToken = null; photo.watermarkWorkExpiresAt = null; 1
        }
        `when`(photos.claimWork(anyArg(), anyArg(), anyArg(), anyArg())).thenReturn(1)

        val claim = assertNotNull(fence().claim(photo.id, PhotoWorkKind.WATERMARK))

        assertNotEquals(stale, claim.token)
        assertEquals(OffsetDateTime.ofInstant(now.minusSeconds(14), ZoneOffset.UTC), fence().endedWatermarkCutoff())
        verify(photos).finishWorkOutput(stale, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))
        verify(photos).finishWork(photo.id, stale)
    }

    @Test fun `claim refuses a held watermark within two storage timeouts and any held indexing token`() {
        photo.watermarkWorkToken = UUID.randomUUID()
        photo.watermarkWorkExpiresAt = OffsetDateTime.ofInstant(now.minusSeconds(14), ZoneOffset.UTC)
        photo.indexingWorkToken = UUID.randomUUID()
        photo.indexingWorkExpiresAt = OffsetDateTime.ofInstant(now.minusSeconds(3600), ZoneOffset.UTC)
        `when`(photos.findAllByIdForUpdate(listOf(photo.id))).thenReturn(listOf(photo))

        assertNull(fence().claim(photo.id, PhotoWorkKind.WATERMARK))
        assertNull(fence().claim(photo.id, PhotoWorkKind.INDEXING))
        verify(photos, never()).finishWorkOutput(anyArg(), anyArg())
        verify(photos, never()).finishWork(anyArg(), anyArg())
        verify(photos, never()).claimWork(anyArg(), anyArg(), anyArg(), anyArg())
    }

    private fun endedIndexing(provider: String = "rekognition", batchId: UUID? = null, cleanupFinished: Boolean = true): UUID {
        val token = UUID.randomUUID()
        photo.indexingWorkToken = token
        photo.indexingWorkExpiresAt = OffsetDateTime.ofInstant(now.minusSeconds(44), ZoneOffset.UTC)
        `when`(photos.findAllByIdForUpdate(listOf(photo.id))).thenReturn(listOf(photo))
        val output = mock(com.quickpitik.repository.PhotoFaceWork::class.java)
        `when`(output.token).thenReturn(token)
        `when`(output.provider).thenReturn(provider)
        `when`(output.batchId).thenReturn(batchId)
        `when`(output.priorCleanupFinishedAt).thenReturn(if (cleanupFinished) now else null)
        `when`(output.expiresAt).thenReturn(now.minusSeconds(44))
        `when`(photos.faceWorkForPhoto(photo.eventId, photo.id)).thenReturn(listOf(output))
        return token
    }

    @Test fun `a paused prior cleanup keeps an expired claim held`() {
        endedIndexing(cleanupFinished = false)
        assertFalse(fence().releaseEndedIndexing(photo.id))
        verify(photos, never()).finishWorkOutput(anyArg(), anyArg())
        verify(photos, never()).finishWork(anyArg(), anyArg())
    }

    @Test fun `an indexing claim is released only after four call deadlines past its expiry`() {
        val token = endedIndexing()
        photo.indexingWorkExpiresAt = OffsetDateTime.ofInstant(now.minusSeconds(43), ZoneOffset.UTC)
        assertFalse(fence().releaseEndedIndexing(photo.id))
        verify(photos, never()).finishWorkOutput(anyArg(), anyArg())

        photo.indexingWorkExpiresAt = OffsetDateTime.ofInstant(now.minusSeconds(44), ZoneOffset.UTC)
        assertTrue(fence().releaseEndedIndexing(photo.id))
        assertEquals(OffsetDateTime.ofInstant(now.minusSeconds(44), ZoneOffset.UTC), fence().endedIndexingCutoff())
        verify(photos).finishWork(photo.id, token)
    }

    @Test fun `releasing an ended indexing claim finishes its output row and clears both columns`() {
        val token = endedIndexing()
        assertTrue(fence().releaseEndedIndexing(photo.id))
        verify(photos).finishWorkOutput(token, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))
        verify(photos).finishWork(photo.id, token)
    }

    @Test fun `an ai_api claim four deadlines past expiry is not released`() {
        endedIndexing("ai_api")
        assertFalse(fence().releaseEndedIndexing(photo.id))
        verify(photos, never()).finishWorkOutput(anyArg(), anyArg())
        verify(photos, never()).finishWork(anyArg(), anyArg())
    }

    @Test fun `a batch claim is held even if the sweep now runs in per photo mode`() {
        endedIndexing(batchId = UUID.randomUUID())
        assertFalse(fence().releaseEndedIndexing(photo.id))
        verify(photos, never()).finishWorkOutput(anyArg(), anyArg())
    }

    @Test fun `batch mode keeps expired Rekognition claims held`() {
        endedIndexing()
        assertFalse(fence(mode = IndexingMode.BATCH).releaseEndedIndexing(photo.id))
        verify(photos, never()).finishWorkOutput(anyArg(), anyArg())
    }

    @Test fun `release requires the matching unfinished output and respects its later deadline`() {
        endedIndexing()
        val output = photos.faceWorkForPhoto(photo.eventId, photo.id).single()
        `when`(output.expiresAt).thenReturn(now.minusSeconds(43))
        assertFalse(fence().releaseEndedIndexing(photo.id))
        `when`(output.expiresAt).thenReturn(now.minusSeconds(44))
        `when`(output.finishedAt).thenReturn(now)
        assertFalse(fence().releaseEndedIndexing(photo.id))
        `when`(output.finishedAt).thenReturn(null)
        photo.indexingWorkToken = UUID.randomUUID()
        assertFalse(fence().releaseEndedIndexing(photo.id))
        verify(photos, never()).finishWorkOutput(anyArg(), anyArg())
    }

    @Test fun `release rechecks restrictions retry state and budget under the photo lock`() {
        endedIndexing()
        photo.deletedAt = OffsetDateTime.ofInstant(now, ZoneOffset.UTC)
        assertFalse(fence().releaseEndedIndexing(photo.id))
        photo.deletedAt = null
        photo.deletionStartedAt = OffsetDateTime.ofInstant(now, ZoneOffset.UTC)
        assertFalse(fence().releaseEndedIndexing(photo.id))
        photo.deletionStartedAt = null
        photo.removalState = RemovalState.REVIEW
        assertFalse(fence().releaseEndedIndexing(photo.id))
        photo.removalState = RemovalState.NONE
        photo.indexingAttempts = 5
        assertFalse(fence().releaseEndedIndexing(photo.id))
        photo.indexingAttempts = 0
        for (status in listOf(com.quickpitik.entity.IndexingStatus.BATCHING, com.quickpitik.entity.IndexingStatus.INDEXED,
            com.quickpitik.entity.IndexingStatus.SKIPPED)) {
            photo.indexingStatus = status
            assertFalse(fence().releaseEndedIndexing(photo.id))
        }
        verify(photos, never()).finishWorkOutput(anyArg(), anyArg())
    }

    @Test fun `Rekognition write settles only four call deadlines after its claim expired`() {
        val early = PhotoWorkClaim(photo.id, 0, UUID.randomUUID(), now.minusSeconds(43), PhotoWorkKind.INDEXING)
        assertFalse(fence().finishEndedRekognitionWrite(early))
        verify(photos, never()).finishWorkOutput(anyArg(), anyArg())

        val ended = PhotoWorkClaim(photo.id, 0, UUID.randomUUID(), now.minusSeconds(44), PhotoWorkKind.INDEXING)
        assertTrue(fence().finishEndedRekognitionWrite(ended))
        verify(photos).finishWorkOutput(ended.token, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))
        verify(photos).finishWork(photo.id, ended.token)
    }
}
