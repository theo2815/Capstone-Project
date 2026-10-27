package com.quickpitik.service.photographer

import com.quickpitik.entity.Photo
import com.quickpitik.repository.PhotoRepository
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.util.UUID
import kotlin.test.assertEquals

// FS-2 (R5-1): the watermark sweep counts the photos the hot path missed.
class PhotoWatermarkTriggerTest {
    private val watermarkService = Mockito.mock(PhotoWatermarkService::class.java)
    private val photoRepo = Mockito.mock(PhotoRepository::class.java)
    private val registry = SimpleMeterRegistry()

    private fun <T> anyArg(): T = Mockito.any()

    private fun photo() = Photo(eventId = UUID.randomUUID(), s3Key = "k", pricePhp = BigDecimal.TEN)

    @Test
    fun `reconcile re-drives and counts the photos it picks up`() {
        val p1 = photo()
        val p2 = photo()
        Mockito.`when`(photoRepo.findWatermarkBacklog(Mockito.anyInt(), anyArg(), anyArg()))
            .thenReturn(listOf(p1, p2), emptyList())
        val trigger = PhotoWatermarkTrigger(watermarkService, photoRepo, 5, registry)

        trigger.reconcile()
        trigger.reconcile()

        Mockito.verify(watermarkService).process(p1.id)
        Mockito.verify(watermarkService).process(p2.id)
        assertEquals(2.0, registry.counter("qp.watermark.reconciled").count())
    }
}
