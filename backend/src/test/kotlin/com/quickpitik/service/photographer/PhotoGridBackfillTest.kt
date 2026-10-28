package com.quickpitik.service.photographer

import com.quickpitik.config.ThumbnailProperties
import com.quickpitik.entity.Photo
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.photos.PhotoWorkClaim
import com.quickpitik.service.photos.PhotoWorkFence
import com.quickpitik.service.photos.PhotoWorkKind
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.Pageable
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Instant
import java.util.UUID
import kotlin.test.assertEquals

// FS-10 (R7b): photos published before FS-9 get their 800 px grid.jpg, built
// from the stored marked preview, only while the owner has the sweep on.
class PhotoGridBackfillTest {
    private val photoRepo = Mockito.mock(PhotoRepository::class.java)
    private val storage = Mockito.mock(StorageService::class.java)
    private val watermarkService = Mockito.mock(WatermarkService::class.java)
    private val workFence = Mockito.mock(PhotoWorkFence::class.java)
    private val transactionManager = Mockito.mock(PlatformTransactionManager::class.java).also {
        Mockito.`when`(it.getTransaction(anyArg())).thenReturn(SimpleTransactionStatus())
    }
    private val preview = byteArrayOf(1, 2, 3)
    private val tile = byteArrayOf(4, 5)

    private fun <T> anyArg(): T = Mockito.any()

    private fun photo() = Photo(eventId = UUID.randomUUID(), s3Key = "original.jpg", pricePhp = BigDecimal.TEN)
        .apply { watermarkS3Key = "events/$eventId/photos/$id/watermark.jpg"; thumbnailS3Key = watermarkS3Key }

    private fun gridKey(p: Photo) = "events/${p.eventId}/photos/${p.id}/grid.jpg"

    private fun backfill(enabled: Boolean = true) = PhotoGridBackfill(
        photoRepo, storage, watermarkService, workFence, TransactionTemplate(transactionManager),
        ThumbnailProperties(backfillEnabled = enabled),
    )

    private fun grantClaims() {
        Mockito.`when`(workFence.claim(anyArg(), anyArg())).thenAnswer {
            PhotoWorkClaim(it.getArgument(0), 0, UUID.randomUUID(), Instant.now().plusSeconds(60), it.getArgument(1))
        }
        Mockito.`when`(workFence.isCurrent(anyArg())).thenReturn(true)
    }

    @Test fun `off by default it never touches the database`() {
        assertEquals(false, ThumbnailProperties().backfillEnabled)
        PhotoGridBackfill(photoRepo, storage, watermarkService, workFence, TransactionTemplate(transactionManager),
            ThumbnailProperties()).backfill()

        Mockito.verifyNoInteractions(photoRepo, storage, workFence)
    }

    @Test fun `one tick writes the tile under a watermark claim and points the row at it`() {
        val p = photo()
        grantClaims()
        Mockito.`when`(photoRepo.findGridBackfill(anyArg(), anyArg())).thenReturn(listOf(p))
        Mockito.`when`(storage.getBytes(p.watermarkS3Key!!)).thenReturn(preview)
        Mockito.`when`(watermarkService.gridFromPreview(preview)).thenReturn(tile)
        Mockito.`when`(photoRepo.setGridKey(p.id, gridKey(p))).thenReturn(1)

        backfill().backfill()

        Mockito.verify(workFence).claim(p.id, PhotoWorkKind.WATERMARK)
        Mockito.verify(storage).put(gridKey(p), tile, "image/jpeg")
        Mockito.verify(photoRepo).setGridKey(p.id, gridKey(p))
        Mockito.verify(workFence).finish(anyArg())
    }

    @Test fun `a photo another worker holds is skipped without a storage call`() {
        val held = photo()
        val free = photo()
        grantClaims()
        Mockito.doReturn(null).`when`(workFence).claim(held.id, PhotoWorkKind.WATERMARK)
        Mockito.`when`(photoRepo.findGridBackfill(anyArg(), anyArg())).thenReturn(listOf(held, free))
        Mockito.`when`(storage.getBytes(free.watermarkS3Key!!)).thenReturn(preview)
        Mockito.`when`(watermarkService.gridFromPreview(preview)).thenReturn(tile)

        backfill().backfill()

        Mockito.verify(storage, Mockito.never()).getBytes(held.watermarkS3Key!!)
        Mockito.verify(storage).put(gridKey(free), tile, "image/jpeg")
    }

    @Test fun `a failed tile upload keeps the claim and leaves the row alone`() {
        val p = photo()
        grantClaims()
        Mockito.`when`(photoRepo.findGridBackfill(anyArg(), anyArg())).thenReturn(listOf(p))
        Mockito.`when`(storage.getBytes(p.watermarkS3Key!!)).thenReturn(preview)
        Mockito.`when`(watermarkService.gridFromPreview(preview)).thenReturn(tile)
        Mockito.`when`(storage.put(gridKey(p), tile, "image/jpeg")).thenThrow(RuntimeException("R2 down"))

        backfill().backfill()

        // The PUT may still land remotely: the claim stays for reconciliation, like PhotoWatermarkService.
        Mockito.verify(workFence, Mockito.never()).finish(anyArg())
        Mockito.verify(photoRepo, Mockito.never()).setGridKey(anyArg(), anyArg())
    }

    @Test fun `it walks forward by id and stops once a page comes back empty`() {
        val a = photo()
        val b = photo()
        grantClaims()
        val afters = mutableListOf<UUID>()
        Mockito.`when`(photoRepo.findGridBackfill(anyArg(), anyArg())).thenAnswer {
            afters += it.getArgument<UUID>(0)
            assertEquals(50, it.getArgument<Pageable>(1).pageSize)
            if (afters.size == 1) listOf(a, b) else emptyList()
        }
        Mockito.`when`(storage.getBytes(anyArg())).thenReturn(preview)
        Mockito.`when`(watermarkService.gridFromPreview(preview)).thenReturn(tile)

        val sweep = backfill()
        repeat(3) { sweep.backfill() }

        // The third tick finds the pass finished and asks for nothing.
        assertEquals(listOf(UUID(0, 0), b.id), afters)
    }
}
