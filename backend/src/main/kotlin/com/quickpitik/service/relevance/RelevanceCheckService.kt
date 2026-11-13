package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.RelevanceCase
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CheckDue
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.service.events.postgresTimestamp
import io.micrometer.core.instrument.MeterRegistry
import io.micrometer.core.instrument.Timer
import org.slf4j.LoggerFactory
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import java.util.concurrent.TimeUnit

// RV-1.5 (plan 5.8): the version 2 check tick. It starts a check for each due batch: claim the batch, read
// again that it is still due, then one transaction inserts the case in SAMPLING, picks its sample and marks
// the batch checked. No provider call here; later tasks add the measure, decide and apply steps.
@Service
class RelevanceCheckService(
    private val batches: BatchRelevanceRepository,
    private val cases: RelevanceCaseRepository,
    private val props: RelevanceProperties,
    private val switch: RelevanceSwitch,
    private val clock: Clock,
    private val transactions: TransactionTemplate,
    private val registry: MeterRegistry,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Scheduled(
        initialDelayString = "\${app.relevance.check.tick-interval-ms:15000}",
        fixedDelayString = "\${app.relevance.check.tick-interval-ms:15000}",
    )
    fun tick() {
        if (switch.mode == RelevanceMode.OFF) return
        // Later tasks add their steps here.
        // G2: with AI off no check starts, so batches stay "not checked".
        if (switch.aiEnabled) startChecks()
    }

    private fun startChecks() {
        val now = clock.postgresTimestamp()
        val quietCutoff = now.minusMinutes(props.quietMinutes)
        val due = batches.findCheckDue(quietCutoff, now, props.check.countTrigger, props.check.maxBatchesPerTick)
        if (due.isEmpty()) return
        val timer = Timer.start(registry)
        var started = 0
        for (batch in due) {
            try {
                if (start(batch, quietCutoff, now)) started++
            } catch (ex: Exception) {
                log.warn("Relevance check start failed event={} photographer={} (retried next tick): {}",
                    batch.eventId, batch.photographerId, ex.message)
            }
        }
        val elapsed = timer.stop(registry.timer("qp.relevance.check.tick"))
        log.info("Relevance check tick due={} started={} ms={}", due.size, started, TimeUnit.NANOSECONDS.toMillis(elapsed))
    }

    private fun start(batch: CheckDue, quietCutoff: OffsetDateTime, now: OffsetDateTime): Boolean {
        val id = batch.batchId ?: transactions.execute {
            batches.ensureBatch(batch.eventId, batch.photographerId)
            batches.findByEventIdAndPhotographerId(batch.eventId, batch.photographerId)?.id
        } ?: return false
        val token = UUID.randomUUID()
        val claimed = transactions.execute { batches.claimCheck(id, token, now.plusMinutes(props.check.claimMinutes), now) }
        if (claimed != 1) return false
        try {
            return transactions.execute { status ->
                // Another copy may have started this check after our due list was read.
                val fresh = batches.findCheckDueFor(id, token, quietCutoff, props.check.countTrigger) ?: return@execute false
                val uploadAt = fresh.lastUploadAt.atOffset(ZoneOffset.UTC)
                val case = cases.saveAndFlush(RelevanceCase(batchId = id, trigger = fresh.trigger, uploadAt = uploadAt,
                    batchPhotos = fresh.photoCount))
                cases.pickSample(case.id, fresh.eventId, fresh.photographerId, 1, props.check.sampleSize)
                if (batches.markChecked(id, token, uploadAt, fresh.photoCount) != 1) {
                    status.setRollbackOnly()
                    return@execute false
                }
                true
            } == true
        } finally {
            transactions.execute { batches.releaseCheck(id, token) }
        }
    }
}
