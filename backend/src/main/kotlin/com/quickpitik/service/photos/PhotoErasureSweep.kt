package com.quickpitik.service.photos

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.dto.photos.PhotoErasureOutcome
import com.quickpitik.repository.PhotoErasureJobRepository
import io.micrometer.core.instrument.MeterRegistry
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component
import java.time.Clock
import java.time.Instant
import java.util.concurrent.atomic.AtomicLong

@Component
class PhotoErasureSweep(
    private val jobs: PhotoErasureJobRepository,
    private val erasure: PhotoErasureService,
    private val properties: PhotoRemovalProperties,
    private val clock: Clock,
    meterRegistry: MeterRegistry,
) {
    private val log = LoggerFactory.getLogger(javaClass)
    private val open = meterRegistry.gauge("qp.removal.erasure.open", AtomicLong())!!

    @Scheduled(fixedDelayString = "\${app.photo-removal.cleanup-interval:PT1M}", initialDelayString = "\${app.photo-removal.cleanup-interval:PT1M}")
    fun scheduled() { sweepDue(clock.instant(), properties.sweepBatchSize) }

    fun sweepDue(now: Instant, limit: Int): Int {
        if (limit <= 0) return 0
        val due = jobs.findDue(now, PageRequest.of(0, minOf(limit, properties.sweepBatchSize)))
        due.forEach { id ->
            runCatching { erasure.attempt(id) }
                .onFailure { log.warn("Erasure attempt deferred for job {} ({})", id, it.javaClass.simpleName) }
        }
        open.set(jobs.countByOutcomeNot(PhotoErasureOutcome.COMPLETE))
        return due.size
    }
}
