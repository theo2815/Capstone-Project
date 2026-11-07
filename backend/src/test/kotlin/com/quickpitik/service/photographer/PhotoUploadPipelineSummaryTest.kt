package com.quickpitik.service.photographer

import com.quickpitik.repository.PhotoRepository
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import java.time.Clock
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class PhotoUploadPipelineSummaryTest {
    private val registry = SimpleMeterRegistry()
    private val uploads = Mockito.mock(PhotoUploadService::class.java)
    private val photos = Mockito.mock(PhotoRepository::class.java)
    private val clock = Clock.fixed(Instant.parse("2026-11-07T00:01:00Z"), ZoneOffset.UTC)
    private val indexing = ThreadPoolTaskExecutor().apply { corePoolSize = 6; initialize() }
    private val watermark = ThreadPoolTaskExecutor().apply { corePoolSize = 6; initialize() }

    private fun summary() = PhotoUploadPipelineSummary(uploads, photos, registry, indexing, watermark, clock)
    private fun minute(commits: Long = 0) = UploadMinute(
        commits = commits, waited = 2, refused = 3, roomPeak = 7, slotsPeak = 4,
        roomCapacity = 16, slotCapacity = 4, meanCommitSeconds = 1.25,
        topEvents = listOf(UUID.fromString("12345678-1234-1234-1234-123456789abc") to 5L),
    )

    @Test
    fun `idle minute logs nothing`() {
        Mockito.`when`(uploads.drainMinute()).thenReturn(minute())
        try {
            assertNull(summary().pipelineLine())
            Mockito.verifyNoInteractions(photos)
        } finally {
            indexing.shutdown()
            watermark.shutdown()
        }
    }

    @Test
    fun `pipeline line has the exact text for a sample minute`() {
        val pool = UploadPool(active = 3, size = 6, queued = 2)
        assertEquals(
            "Upload pipeline, last 60 s: commits 5 (waited 2, refused 3), slot mean 1250 ms, " +
                "room peak 7/16, slots peak 4/4; watermark active 3/6 queued 2 oldest PROCESSING 40s; " +
                "indexing active 3/6 queued 2 oldest PENDING 20s; rekognition errors 8 (throttled 5); " +
                "top events 12345678 5",
            formatPipelineLine(minute(5), pool, pool, 40, 20, 8, 5),
        )
        indexing.shutdown()
        watermark.shutdown()
    }

    @Test
    fun `throttled errors are counted apart from other errors and reset each minute`() {
        Mockito.`when`(uploads.drainMinute()).thenReturn(minute(5))
        Mockito.`when`(photos.findOldestProcessingUpload()).thenReturn(OffsetDateTime.now(clock).minusSeconds(40))
        Mockito.`when`(photos.findOldestPendingIndexingUpload()).thenReturn(OffsetDateTime.now(clock).minusSeconds(20))
        val summary = summary()
        registry.counter("qp.ai.error", "op", "index", "code", "ThrottlingException").increment(2.0)
        registry.counter("qp.ai.error", "op", "ocr", "code", "ProvisionedThroughputExceededException").increment(3.0)
        registry.counter("qp.ai.error", "op", "search", "code", "AccessDeniedException").increment(3.0)
        try {
            val line = checkNotNull(summary.pipelineLine())
            assertTrue(line.contains("rekognition errors 8 (throttled 5)"), line)
            assertTrue(line.contains("oldest PROCESSING 40s"), line)
            assertTrue(line.contains("oldest PENDING 20s"), line)
            assertTrue(checkNotNull(summary.pipelineLine()).contains("rekognition errors 0 (throttled 0)"))
        } finally {
            indexing.shutdown()
            watermark.shutdown()
        }
    }

    @Test
    fun `queue size of a real ThreadPoolTaskExecutor is read`() {
        val entered = CountDownLatch(6)
        val release = CountDownLatch(1)
        try {
            repeat(6) { indexing.execute { entered.countDown(); check(release.await(5, TimeUnit.SECONDS)) } }
            assertTrue(entered.await(2, TimeUnit.SECONDS))
            indexing.execute { }
            indexing.execute { }
            Mockito.`when`(uploads.drainMinute()).thenReturn(minute())
            val line = checkNotNull(summary().pipelineLine())
            assertTrue(line.contains("indexing active 6/6 queued 2"), line)
        } finally {
            release.countDown()
            indexing.shutdown()
            watermark.shutdown()
        }
    }
}
