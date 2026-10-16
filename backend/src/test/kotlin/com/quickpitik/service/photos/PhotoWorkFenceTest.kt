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
    private fun fence(provider: AiProvider = AiProvider.REKOGNITION) = PhotoWorkFence(photos,
        StorageProperties(operationTimeout = Duration.ofSeconds(7)), AiApiProperties(), AiProperties(provider = provider),
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
}
