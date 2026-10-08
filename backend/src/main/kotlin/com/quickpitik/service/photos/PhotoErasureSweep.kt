package com.quickpitik.service.photos

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.repository.PhotoErasureJobRepository
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component
import java.time.Clock
import java.time.Instant

@Component
class PhotoErasureSweep(
    private val jobs: PhotoErasureJobRepository,
    private val erasure: PhotoErasureService,
    private val properties: PhotoRemovalProperties,
    private val clock: Clock,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Scheduled(fixedDelayString = "\${app.photo-removal.cleanup-interval:PT1M}", initialDelayString = "\${app.photo-removal.cleanup-interval:PT1M}")
    fun scheduled() { sweepDue(clock.instant(), properties.sweepBatchSize) }

    fun sweepDue(now: Instant, limit: Int): Int {
        if (limit <= 0) return 0
        val due = jobs.findDue(now, PageRequest.of(0, minOf(limit, properties.sweepBatchSize)))
        due.forEach { id ->
            runCatching { erasure.attempt(id) }
                .onFailure { log.warn("Erasure attempt deferred for job {} ({})", id, it.javaClass.simpleName) }
        }
        return due.size
    }
}
