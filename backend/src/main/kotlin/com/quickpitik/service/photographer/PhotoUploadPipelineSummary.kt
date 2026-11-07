package com.quickpitik.service.photographer

import com.quickpitik.repository.PhotoRepository
import io.micrometer.core.instrument.MeterRegistry
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.core.task.TaskExecutor
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import org.springframework.stereotype.Component
import java.time.Clock
import java.time.Duration
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.math.roundToLong

data class UploadMinute(
    val commits: Long,
    val waited: Long,
    val refused: Long,
    val roomPeak: Int,
    val slotsPeak: Int,
    val roomCapacity: Int,
    val slotCapacity: Int,
    val meanCommitSeconds: Double,
    val topEvents: List<Pair<UUID, Long>>,
)

internal data class UploadPool(val active: Int, val size: Int, val queued: Int)

@Component
class PhotoUploadPipelineSummary(
    private val uploads: PhotoUploadService,
    private val photos: PhotoRepository,
    private val registry: MeterRegistry,
    @Qualifier("imageProcessing") private val indexing: TaskExecutor,
    @Qualifier("watermarkProcessing") private val watermark: TaskExecutor,
    private val clock: Clock,
) {
    private val log = LoggerFactory.getLogger(javaClass)
    private var previousErrors = 0.0
    private var previousThrottled = 0.0

    @Scheduled(fixedDelayString = "\${app.storage.direct-upload.summary-interval:PT1M}")
    fun summarize() {
        try {
            pipelineLine()?.let { log.info(it) }
        } catch (error: Exception) {
            log.warn("Upload pipeline summary failed error={}", error.javaClass.simpleName)
        }
    }

    internal fun pipelineLine(): String? {
        val minute = uploads.drainMinute()
        val watermarkPool = pool(watermark)
        val indexingPool = pool(indexing)
        val counters = registry.find("qp.ai.error").counters()
        val errors = counters.sumOf { it.count() }
        val throttled = counters.filter {
            it.id.getTag("code") in setOf("ThrottlingException", "ProvisionedThroughputExceededException")
        }.sumOf { it.count() }
        val errorDelta = (errors - previousErrors).toLong()
        val throttledDelta = (throttled - previousThrottled).toLong()
        previousErrors = errors
        previousThrottled = throttled
        if (minute.commits == 0L && watermarkPool.queued == 0 && indexingPool.queued == 0) return null
        return formatPipelineLine(minute, watermarkPool, indexingPool,
            age(photos.findOldestProcessingUpload()), age(photos.findOldestPendingIndexingUpload()),
            errorDelta, throttledDelta)
    }

    private fun pool(executor: TaskExecutor): UploadPool {
        val pool = executor as? ThreadPoolTaskExecutor ?: return UploadPool(0, 0, 0)
        return UploadPool(pool.activeCount, pool.corePoolSize, pool.threadPoolExecutor.queue.size)
    }

    private fun age(uploadedAt: OffsetDateTime?): Long = uploadedAt?.let {
        Duration.between(it, OffsetDateTime.now(clock)).seconds.coerceAtLeast(0)
    } ?: 0
}

internal fun formatPipelineLine(
    minute: UploadMinute, watermark: UploadPool, indexing: UploadPool,
    processingAge: Long, pendingAge: Long, errors: Long, throttled: Long,
): String = "Upload pipeline, last 60 s: commits ${minute.commits} (waited ${minute.waited}, refused ${minute.refused}), " +
    "slot mean ${(minute.meanCommitSeconds * 1000).roundToLong()} ms, " +
    "room peak ${minute.roomPeak}/${minute.roomCapacity}, slots peak ${minute.slotsPeak}/${minute.slotCapacity}; " +
    "watermark active ${watermark.active}/${watermark.size} queued ${watermark.queued} oldest PROCESSING ${processingAge}s; " +
    "indexing active ${indexing.active}/${indexing.size} queued ${indexing.queued} oldest PENDING ${pendingAge}s; " +
    "rekognition errors $errors (throttled $throttled); top events " +
    minute.topEvents.joinToString(", ") { "${it.first.toString().take(8)} ${it.second}" }
