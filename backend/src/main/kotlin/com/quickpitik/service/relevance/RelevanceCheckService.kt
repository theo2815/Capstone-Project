package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.BatchRelevance
import com.quickpitik.entity.CaseBand
import com.quickpitik.entity.CaseStage
import com.quickpitik.entity.Event
import com.quickpitik.entity.RelevanceCase
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CheckDue
import com.quickpitik.repository.EventRepository
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
// moves the case to DECIDING with its counts and band. RV-1.8b: it decides each DECIDING case with one model call
// (no transaction open), lets code check the answer (settle) and stores the verdict. Later tasks add the apply step.
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
    private val client: AnthropicClient,
    private val events: EventRepository,
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
        // O6: no model call without AI on and a key.
        if (switch.canDecide) decideCases()
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
    internal fun continueCase(row: SamplingCase, now: OffsetDateTime) {
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
        // A RULES re-check decides on the sample it copied: it never pays for a second round (P9).
        if (case.sampleRound.toInt() == 1 && case.trigger != RULES && band != CaseBand.SMALL && band != CaseBand.UNKNOWN &&
            nearDumpLine(counts, props)) {
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

    internal fun start(batch: CheckDue, quietCutoff: OffsetDateTime, now: OffsetDateTime): Boolean {
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
                // RULES (P9): the current case's sample and round, so the re-check costs one model call and no AWS call.
                val current = if (fresh.trigger == RULES) cases.currentFor(id) ?: return@execute false else null
                val case = cases.saveAndFlush(RelevanceCase(batchId = id, trigger = fresh.trigger, uploadAt = uploadAt,
                    batchPhotos = fresh.photoCount, sampleRound = current?.sampleRound ?: 1))
                if (current != null) cases.copySample(case.id, current.id)
                else cases.pickSample(case.id, fresh.eventId, fresh.photographerId, 1, props.check.sampleSize)
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

    private fun decideCases() {
        val now = clock.postgresTimestamp()
        for (row in cases.findDeciding(now, now.minusMinutes(props.check.waitForRulesMinutes), props.check.maxBatchesPerTick)) {
            try {
                decideCase(row, now)
            } catch (ex: Exception) {
                log.warn("Relevance decide step failed case={} (retried next tick): {}", row.caseId, ex.message)
            }
        }
    }

    // Plan 5.9: one model call for one DECIDING case, made under the batch claim with no transaction open. Code
    // settles the answer; the verdict and everything that explains it is one short write guarded by claim and stage.
    private fun decideCase(row: SamplingCase, now: OffsetDateTime) {
        val token = UUID.randomUUID()
        val claimed = transactions.execute { batches.claimCheck(row.batchId, token, now.plusMinutes(props.check.claimMinutes), now) }
        if (claimed != 1) return
        try {
            val case = cases.findById(row.caseId).orElse(null)
            if (case?.stage != CaseStage.DECIDING.name) return
            if (!batches.stillAllowed(row.batchId)) {
                transactions.execute { cases.closeSkippedDeciding(row.caseId, token, now) }
                log.info("Relevance check skipped case={}: event not public or photographer suspended", row.caseId)
                return
            }
            val batch = batches.findById(row.batchId).orElse(null) ?: return
            val sameUpload = batch.scoredUploadAt?.isEqual(case.uploadAt) == true
            val waited = case.createdAt?.let { !it.isAfter(now.minusMinutes(props.check.waitForRulesMinutes)) } == true
            if (!sameUpload && !waited) return
            val event = events.findById(row.eventId).orElse(null) ?: return
            val sharedRace = event.createdBy == null
            val rules = ruleResults(batch, sameUpload, sharedRace, props.date.minDatedPhotos)
            val wordsPhotos = batches.countWordPhotos(row.batchId).toInt()
            val summary = RelevanceSummary.build(summaryInput(case, row.eventId, event, sharedRace, rules, wordsPhotos))

            var failure: String? = null
            var answer: ModelAnswer? = null
            var model = props.llm.model
            // A crash after the raise and before the write costs at most one more call (R5).
            if (case.decideAttempts >= props.check.decideMaxAttempts) {
                failure = "CALL_FAILED"
            } else {
                if (transactions.execute { cases.raiseDecideAttempts(case.id, token) } != 1) return
                when (val result = client.ask(PROMPT_RV1, summary, props.llm.maxTokens)) {
                    is LlmResult.Answer -> {
                        model = result.model
                        answer = RelevanceSummary.parseAnswer(result.text)
                        if (answer == null) failure = "BAD_JSON"
                    }
                    LlmResult.CutOff -> failure = "CUT_OFF"
                    is LlmResult.Failed -> {
                        if (case.decideAttempts + 1 < props.check.decideMaxAttempts) {
                            log.info("Relevance verdict call failed case={} attempt={} (tried again next tick): {}",
                                case.id, case.decideAttempts + 1, result.reason)
                            return
                        }
                        failure = "CALL_FAILED"
                    }
                }
            }
            val band = case.band?.let { CaseBand.valueOf(it) } ?: CaseBand.UNKNOWN
            val settled = settle(answer, failure, band, rules.copyLow, rules.dateLow, sharedRace, wordsPhotos, case.batchPhotos)
            val written = transactions.execute {
                cases.saveVerdict(case.id, token, now, settled.verdict.name, settled.note, settled.modelVerdict?.name,
                    answer?.adminReason, answer?.evidenceUsed?.joinToString(","), summary, model.take(MODEL_COLUMN), PROMPT_VERSION,
                    rules.copyLow, rules.dateLow, wordsPhotos)
            }
            if (written != 1) return
            registry.counter("qp.relevance.decided", "verdict", settled.verdict.name.lowercase()).increment()
            log.info("Relevance check decided case={} verdict={} note={}", case.id, settled.verdict, settled.note)
        } finally {
            transactions.execute { batches.releaseCheck(row.batchId, token) }
        }
    }

    private fun summaryInput(case: RelevanceCase, eventId: UUID, event: Event, sharedRace: Boolean, rules: RuleResults, wordsPhotos: Int): SummaryInput {
        val share = props.words.minSharePercent
        val batchWords = batches.topWords(case.batchId, share, props.words.topN).associate { it.word to it.photos }
        val eventWords = if (batchWords.isEmpty()) emptyMap() else
            batches.eventWords(eventId, case.batchId, batchWords.keys.toTypedArray(), share).associate { it.word to it.photographers }
        return SummaryInput(
            caseId = case.id, eventName = event.name, eventDate = event.date, place = event.location, sharedRace = sharedRace,
            photos = case.batchPhotos,
            sample = SampleCounts(case.measured ?: 0, case.blurry ?: 0, case.dark ?: 0, case.notRace ?: 0, case.junk ?: 0),
            band = case.band?.let { CaseBand.valueOf(it) } ?: CaseBand.UNKNOWN,
            copyRule = rules.copyRule, dateRule = rules.dateRule, wordsFromPhotos = wordsPhotos, batchWords = batchWords,
            otherPhotographers = batches.otherPhotographers(eventId, case.batchId), eventWords = eventWords,
        )
    }

    private companion object {
        const val RULES = "RULES"
        const val MODEL_COLUMN = 60
    }
}

// What the model is told of version 1's copy and date results, and what the case keeps of them (plan 5.9, P9). A
// result is used only when version 1 scored the case's own upload; otherwise the case keeps null, so the RULES
// trigger re-checks the batch once version 1 has scored it.
internal data class RuleResults(val copyRule: CopyRule, val dateRule: DateRule, val copyLow: Boolean?, val dateLow: Boolean?)

internal fun ruleResults(batch: BatchRelevance, sameUpload: Boolean, sharedRace: Boolean, minDatedPhotos: Int): RuleResults {
    if (!sameUpload) {
        return RuleResults(if (batch.scoredAt == null) CopyRule.NOT_RUN else CopyRule.NOT_FINISHED, DateRule.NOT_RUN, null, null)
    }
    val date = when {
        !sharedRace -> DateRule.NOT_RUN
        batch.dateLow == true -> DateRule.LOW
        // Null while camera dates are still being read; false also below the dated-photo minimum (D21).
        batch.dateLow == null || (batch.datedPhotos ?: 0) < minDatedPhotos -> DateRule.NOT_ENOUGH_DATES
        else -> DateRule.CLEAR
    }
    return RuleResults(if (batch.copyLow == true) CopyRule.LOW else CopyRule.CLEAR, date, batch.copyLow, batch.dateLow)
}
