package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.CaseBand
import com.quickpitik.entity.CaseStage
import com.quickpitik.entity.RelevanceCase
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CheckDue
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.SamplingCase
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
// the batch checked. RV-1.6: it measures each SAMPLING case's sample (RelevanceMeasurer, on its own pool) and
// moves the case to DECIDING with its counts and band. Later tasks add the decide and apply steps.
@Service
class RelevanceCheckService(
    private val batches: BatchRelevanceRepository,
    private val cases: RelevanceCaseRepository,
    private val props: RelevanceProperties,
    private val switch: RelevanceSwitch,
    private val clock: Clock,
    private val transactions: TransactionTemplate,
    private val registry: MeterRegistry,
    private val measurer: RelevanceMeasurer,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Scheduled(
        initialDelayString = "\${app.relevance.check.tick-interval-ms:15000}",
        fixedDelayString = "\${app.relevance.check.tick-interval-ms:15000}",
    )
    fun tick() {
        if (switch.mode == RelevanceMode.OFF) return
        // Later tasks add their steps here.
        // G2: with AI off nothing is measured and no check starts, so batches stay "not checked".
        if (switch.canMeasure) continueSampling()
        if (switch.aiEnabled) startChecks()
    }

    private fun continueSampling() {
        val now = clock.postgresTimestamp()
        for (row in cases.findSampling(now, props.check.maxBatchesPerTick)) {
            try {
                continueCase(row, now)
            } catch (ex: Exception) {
                log.warn("Relevance sample step failed case={} (retried next tick): {}", row.caseId, ex.message)
            }
        }
    }

    // One tick of one sampling case: measure what is left of its sample, or, when nothing is left, store its
    // counts and band (or take the second round near the dump line, G5).
    private fun continueCase(row: SamplingCase, now: OffsetDateTime) {
        val token = UUID.randomUUID()
        val claimed = transactions.execute { batches.claimCheck(row.batchId, token, now.plusMinutes(props.check.claimMinutes), now) }
        if (claimed != 1) return
        try {
            // Another copy may have moved the case on after our list was read.
            val case = cases.findById(row.caseId).orElse(null)
            if (case?.stage != CaseStage.SAMPLING.name) return
            if (!batches.stillAllowed(row.batchId)) {
                transactions.execute { cases.closeSkipped(row.caseId, token, now) }
                log.info("Relevance check skipped case={}: event not public or photographer suspended", row.caseId)
                return
            }
            val todo = batches.nextToMeasure(row.caseId, props.check.measuresPerTick)
            if (todo.isNotEmpty()) {
                measurer.measure(row.batchId, token, todo)
                return
            }
            finishSample(case, row, token)
        } finally {
            transactions.execute { batches.releaseCheck(row.batchId, token) }
        }
    }

    private fun finishSample(case: RelevanceCase, row: SamplingCase, token: UUID) {
        val scores = batches.sampleScores(row.caseId).map {
            PhotoScores(it.fgSharpness, it.fgBrightness, it.sharpness, it.brightness, it.personConf, it.raceConf, it.measureError)
        }
        val counts = sampleCounts(scores, props)
        val band = band(case.batchPhotos, scores.size, counts, props)
        if (case.sampleRound.toInt() == 1 && band != CaseBand.SMALL && band != CaseBand.UNKNOWN && nearDumpLine(counts, props)) {
            val picked = transactions.execute {
                if (cases.startRoundTwo(row.caseId, token) != 1) return@execute -1
                cases.pickSample(row.caseId, row.eventId, row.photographerId, 2, props.check.sampleSize)
            }
            // -1: the claim or the stage was lost. 0: the whole batch is already in round one, so decide on it.
            if (picked != 0) return
        }
        transactions.execute {
            cases.finishSampling(row.caseId, token, scores.size, counts.measured, counts.blurry, counts.dark, counts.notRace,
                counts.junk, band.name)
        }
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
