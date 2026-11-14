package com.quickpitik.service.relevance

import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.AlertKind
import com.quickpitik.entity.BatchRelevance
import com.quickpitik.entity.CaseAction
import com.quickpitik.entity.CaseBand
import com.quickpitik.entity.CaseStage
import com.quickpitik.entity.CaseVerdict
import com.quickpitik.entity.Event
import com.quickpitik.entity.RelevanceCase
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CheckDue
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.HoldTarget
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.repository.SamplingCase
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.auth.atLeast
import com.quickpitik.service.events.postgresTimestamp
import com.quickpitik.websocket.AdminInboxEvent
import io.micrometer.core.instrument.MeterRegistry
import io.micrometer.core.instrument.Timer
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.context.ApplicationEventPublisher
import org.springframework.core.task.TaskExecutor
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import org.springframework.transaction.TransactionStatus
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
// (no transaction open), lets code check the answer (settle) and stores the verdict. RV-1.9: it applies each DECIDED
// case in record-only (applyCase): the case becomes current and stores the alerts auto-hide would have raised.
// RV-2.2a: in auto-hide, with the photographer's consent, the case becomes current with its hold target (A.3).
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
    private val alerts: RelevanceAlertRepository,
    private val publisher: ApplicationEventPublisher,
    private val legal: LegalProperties,
    private val holds: RelevanceHoldRepository,
    private val photos: PhotoRepository,
    private val decisionLog: AdminDecisionLogService,
    // Read only: the upload-indexing pool the full scan yields to (A.4, R6).
    @Qualifier("imageProcessing") private val imageProcessing: TaskExecutor,
) {
    private val log = LoggerFactory.getLogger(javaClass)
    @Volatile private var unreadablePoolLogged = false

    // Part 2 A.4, RV-2.3a: the full scan of a Fits or Mixed auto-hide case. No cursor: each turn measures up to
    // photos-per-turn photos of the batch that have no measurement yet, so later uploads are picked up and a photo that
    // could not be measured never blocks the finish. Only in auto-hide (P14), only cases applied in auto-hide (P18).
    internal fun scanTurns() {
        if (switch.mode != RelevanceMode.AUTO_HIDE || !switch.canMeasure) return
        // R6: yield to upload indexing on this server copy, which shares the processor, memory and downloads.
        // A pool that cannot be read counts as busy, so a wrapped or replaced bean never lets the scan crowd out uploads.
        val pool = imageProcessing as? ThreadPoolTaskExecutor
        if (pool == null && !unreadablePoolLogged) {
            unreadablePoolLogged = true
            log.warn("Relevance scan cannot read the imageProcessing pool ({}); scanning is paused", imageProcessing.javaClass.name)
        }
        val load = pool?.let { it.activeCount + it.threadPoolExecutor.queue.size } ?: Int.MAX_VALUE
        if (load > props.scan.yieldAbove) {
            registry.counter("qp.relevance.scan.yielded").increment()
            return
        }
        val now = clock.postgresTimestamp()
        for (row in cases.findScans(props.scan.maxBatchesPerTick)) {
            try {
                scanTurn(row, now)
            } catch (ex: Exception) {
                log.warn("Relevance scan step failed case={} (retried next tick): {}", row.caseId, ex.message)
            }
        }
    }

    // One turn under the batch claim. The case is read again (and stamped for the rotation) after the claim; the
    // measuring runs with no transaction open; the finish counts the whole batch in one aggregate.
    private fun scanTurn(row: SamplingCase, now: OffsetDateTime) {
        val blur = props.junk.blurCutoff ?: return
        val dark = props.junk.darkCutoff ?: return
        val token = UUID.randomUUID()
        val claimed = transactions.execute { batches.claimCheck(row.batchId, token, now.plusMinutes(props.check.claimMinutes), now) }
        if (claimed != 1) return
        try {
            // The event may have turned Private after the list was read: skip the turn, pay nothing, close nothing (G9).
            if (!batches.stillAllowed(row.batchId)) return
            if (transactions.execute { cases.startScanTurn(row.caseId, now) } != 1) return
            val todo = batches.nextUnmeasured(row.eventId, row.photographerId, props.scan.photosPerTurn)
            if (todo.isNotEmpty()) {
                val measured = measurer.measure(row.batchId, token, todo)
                registry.counter("qp.relevance.scan.measured").increment(measured.toDouble())
                return
            }
            val counts = holds.scanCounts(row.batchId, blur, dark, props.junk.labelMinConfidence)
            val flipped = transactions.execute {
                if (cases.finishScan(row.caseId, counts.measured.toInt(), counts.junk.toInt(), counts.blurry.toInt(), now) != 1) return@execute null
                // A.4 (G5): the finished scan may switch Fits and Mixed once; the hold step then hides or restores.
                cases.flipAfterScan(row.caseId, counts.measured.toInt(), counts.junk.toInt(), props.junk.fitsBelowPercent, now) == 1
            } ?: return
            log.info("Relevance scan finished case={} measured={} junk={} blurry={} flipped={}",
                row.caseId, counts.measured, counts.junk, counts.blurry, flipped)
        } finally {
            transactions.execute { batches.releaseCheck(row.batchId, token) }
        }
    }

    @Scheduled(
        initialDelayString = "\${app.relevance.check.tick-interval-ms:15000}",
        fixedDelayString = "\${app.relevance.check.tick-interval-ms:15000}",
    )
    fun tick() {
        if (switch.mode == RelevanceMode.OFF) return
        // Database only, so it runs with AI off (G15).
        applyDecided()
        // Database only too. Record-only never hides (P18), so the holds follow a target only in auto-hide.
        if (switch.mode == RelevanceMode.AUTO_HIDE) reconcileHolds()
        // G2: with AI off nothing is measured and no check starts, so batches stay "not checked".
        if (switch.canMeasure) continueSampling()
        // O6: no model call without AI on and a key.
        if (switch.canDecide) decideCases()
        if (switch.aiEnabled) startChecks()
        // Last, because it waits for every photo it measures and would delay every other step; samples run before scans
        // (A.4). scanTurns checks the mode and canMeasure itself.
        scanTurns()
    }

    private fun applyDecided() {
        val now = clock.postgresTimestamp()
        for (case in cases.findDecided(props.check.maxBatchesPerTick)) {
            try {
                applyCase(case.id, now)
            } catch (ex: Exception) {
                log.warn("Relevance apply step failed case={} (retried next tick): {}", case.id, ex.message)
            }
        }
    }

    // Plan 5.11 and part 2 A.3: one decided case in one short transaction, database only. In record-only (and in
    // auto-hide for a photographer without consent, NO_CONSENT) the action and the alerts auto-hide would have raised
    // are stored. In auto-hide the case becomes current with its hold target and, for Fits and Mixed, a queued scan;
    // no photo, hold, flag, message or e-mail is touched here (the hide step is RV-2.2b). Every write is guarded by the
    // stage it expects, so a second server copy that read the same case changes nothing.
    internal fun applyCase(caseId: UUID, now: OffsetDateTime) {
        val raised = transactions.execute { status -> applyInTransaction(caseId, now, status) } ?: return
        raised.forEach { registry.counter("qp.relevance.alerts", "kind", it.name.lowercase()).increment() }
    }

    private fun applyInTransaction(caseId: UUID, now: OffsetDateTime, status: TransactionStatus): List<AlertKind> {
        val case = cases.findById(caseId).orElse(null)
        if (case?.stage != CaseStage.DECIDED.name) return emptyList()
        val decided = CaseStage.DECIDED.name
        if (!batches.stillAllowed(case.batchId)) {
            cases.closeCase(case.id, decided, "SKIPPED", now)
            log.info("Relevance check skipped case={}: event not public or photographer suspended", case.id)
            return emptyList()
        }
        val verdict = CaseVerdict.valueOf(case.verdict!!)
        val autoHide = switch.mode == RelevanceMode.AUTO_HIDE
        // P4: auto-hide acts only for a photographer who accepted the minimum Terms; unset means nobody has.
        val consented = autoHide && legal.minVersionOrNull()?.let { atLeast(batches.photographerTermsVersion(case.batchId), it) } == true
        val mode = if (consented) RelevanceMode.AUTO_HIDE else RelevanceMode.RECORD_ONLY
        val current = cases.currentFor(case.batchId)
        // A.3 (P21): with no current case, an admin's last decision on the batch is what the new case is compared with.
        val comparator = current ?: if (autoHide) cases.lastClosedByAdmin(case.batchId) else null
        if (comparator != null) {
            // A failed check changes nothing; a Can't tell never lifts a real hide; the same verdict again is one alert
            // per verdict change, not per check (R7, P8). Each hands its copy and date results to the current case so
            // the RULES trigger does not fire again for them (a failed re-check too: owner call, 2026-11-14).
            val reason = when {
                verdict == CaseVerdict.CANT_TELL && case.verdictNote in FAILED_CALL -> "NO_RESULT"
                verdict == CaseVerdict.CANT_TELL && current?.appliedMode == RelevanceMode.AUTO_HIDE.name &&
                    current.action != CaseAction.NONE.name -> "HELD_BY_CASE"
                // Only a record-only case is replaced by the same verdict, and only by one that can act (P18).
                comparator.verdict == case.verdict ->
                    if (current == null || current.appliedMode == mode.name || current.appliedMode == RelevanceMode.AUTO_HIDE.name) "SAME_VERDICT" else null
                current == null || current.appliedMode != RelevanceMode.AUTO_HIDE.name -> null
                // A record-only case never acts, and closing an acting case is an act (P18; conductor's call, A.3 leaves it open).
                mode == RelevanceMode.RECORD_ONLY -> "HELD_BY_CASE"
                // G7: only an admin or a won appeal lifts a wrong-event or copied hide; a copied finding may replace a wrong-event one.
                current.verdict in NEVER_LIFTED && !(current.verdict == CaseVerdict.WRONG_EVENT.name && verdict == CaseVerdict.COPIED) ->
                    "HELD_BY_CASE"
                else -> null
            }
            if (reason != null) {
                if (cases.closeCase(case.id, decided, reason, now) == 1 && current != null) {
                    cases.copyRuleResults(current.id, case.copyLow, case.dateLow)
                    // P8: the new photos of a Fits or Mixed batch are scanned too (the query skips a record-only case).
                    if (reason == "SAME_VERDICT" && verdict in SCANNED) cases.queueScan(current.id)
                }
                log.info("Relevance check closed case={} reason={}", case.id, reason)
                return emptyList()
            }
        }
        // G7: a Fits or Mixed re-check lifts a current auto-hide junk dump to HIDE_JUNK, so the junk stays hidden photo by
        // photo; the lifted case does not count (nothing confirms a closed case).
        val lifted = current?.appliedMode == RelevanceMode.AUTO_HIDE.name && current.verdict == CaseVerdict.JUNK_DUMP.name &&
            verdict in SCANNED
        val action = if (lifted) CaseAction.HIDE_JUNK else when (verdict) {
            CaseVerdict.MIXED -> CaseAction.HIDE_JUNK
            CaseVerdict.JUNK_DUMP, CaseVerdict.WRONG_EVENT, CaseVerdict.COPIED -> CaseAction.HIDE_ALL
            CaseVerdict.FITS, CaseVerdict.CANT_TELL -> CaseAction.NONE
        }
        val closeReason = if (lifted) "RECHECK_LIFTED" else "SUPERSEDED"
        if ((current != null && cases.closeCase(current.id, CaseStage.APPLIED.name, closeReason, now) != 1) ||
            cases.applyCase(case.id, mode.name, action.name, now) != 1) {
            status.setRollbackOnly()
            return emptyList()
        }
        if (autoHide && !consented) cases.markNoConsent(case.id)
        if (consented && verdict in SCANNED) cases.queueScan(case.id)
        log.info("Relevance check applied case={} verdict={} action={} mode={}", case.id, verdict, action, mode)
        // In auto-hide only the copied alert is raised here; the big-hide alert waits for the photos actually hidden.
        val raise = if (consented) listOfNotNull((AlertKind.COPIED to case.batchPhotos).takeIf { verdict == CaseVerdict.COPIED })
            else wouldHaveAlerts(case, verdict, action)
        return raise.filter { (kind, photos) ->
            raiseAlert(case.batchId, case.id, kind, photos, mode == RelevanceMode.RECORD_ONLY, now)
        }.map { it.first }
    }

    // The alerts of plan 5.11 with the photos the hide would take: the whole batch, or the sample's junk share of it.
    private fun wouldHaveAlerts(case: RelevanceCase, verdict: CaseVerdict, action: CaseAction): List<Pair<AlertKind, Int>> {
        val measured = case.measured ?: 0
        val photos = when (action) {
            CaseAction.HIDE_ALL -> case.batchPhotos
            CaseAction.HIDE_JUNK -> if (measured == 0) 0 else ((case.junk ?: 0).toLong() * case.batchPhotos / measured).toInt()
            CaseAction.NONE -> 0
        }
        return listOfNotNull(
            (AlertKind.BIG_HIDE to photos).takeIf { action != CaseAction.NONE && photos >= props.alerts.bigHidePhotos },
            (AlertKind.COPIED to photos).takeIf { verdict == CaseVerdict.COPIED },
        )
    }

    // Part 2 A.2, RV-2.2b: a current auto-hide case with target HIDE_ALL holds and hides every photo of its batch, a
    // chunk at a time, so photos uploaded after the hide join it on the next tick (R7). Only cases applied in auto-hide
    // are read (P18). Going Private or a suspension later lifts nothing (G9), so "still allowed?" is not checked here.
    // shortcut: every current whole-batch auto-hide case is visited each tick, and each visit is one transaction whose
    // hide chunk is an anti-join over all the batch's photos, even when nothing is left; skip a finished case until a new
    // upload arrives, or rotate with a LIMIT, once whole-batch cases pile up or batches reach tens of thousands of photos.
    // RV-2.3b: target HIDE_JUNK holds and hides each photo measured as junk as the scan measures it, and the restore step
    // brings back what a lowered target no longer covers: under HIDE_JUNK a measured photo that is not junk, under NONE
    // every automatic hold. The restore is level-triggered: it reads the batch's holds, whichever case they name, every
    // tick until none is left, so a hide chunk that committed just after its case stopped being current is undone too.
    // shortcut: every current Mixed auto-hide case costs a junk anti-join and a restore query each tick, as the whole-batch
    // hide does; bound it the same way (skip until a new measurement or upload, or rotate with a LIMIT) when they pile up.
    // shortcut: findHoldTargets probes each current auto-hide NONE case for a live AUTO hold every tick; the only index is on
    // batch_id, so a "no" reads every hold row of the batch, restored ones too (a Mixed batch scanned down to Fits pays for
    // all its restored rows each tick). Add a partial index WHERE restored_at IS NULL by migration, or skip a NONE case
    // until a new hold exists, once such batches reach tens of thousands of restored holds.
    internal fun reconcileHolds() {
        val now = clock.postgresTimestamp()
        val blur = props.junk.blurCutoff
        val dark = props.junk.darkCutoff
        val labelMin = props.junk.labelMinConfidence
        for (target in cases.findHoldTargets()) {
            try {
                when (target.action) {
                    CaseAction.HIDE_ALL.name -> hideBatch(target, now)
                    // The predicate never runs with a cut-off unset (auto-hide cannot be on without both).
                    CaseAction.HIDE_JUNK.name -> if (blur != null && dark != null) {
                        val done = inChunks(target) { hide(holds.hideJunkChunk(target.batchId, target.caseId, target.eventId,
                            target.photographerId, now, props.hide.chunkSize, blur, dark, labelMin)) }
                        // RV-2.4: a junk-only hide is whole once the scan has finished and nothing junk is left to hold.
                        if (done && target.scanFinished && !target.hideFinished) finishHide(target, now, skipEmpty = true)
                        inChunks(target) { restore(holds.restoreNotJunkChunk(target.batchId, now, props.hide.chunkSize, blur, dark, labelMin)) }
                    }
                    else -> inChunks(target) { restore(holds.restoreChunk(target.batchId, AUTO, AUTO, now, props.hide.chunkSize)) }
                }
            } catch (ex: Exception) {
                log.warn("Relevance hold step failed case={} (retried next tick): {}", target.caseId, ex.message)
            }
        }
        finishAdminRestores(now)
    }

    // RV-2.5a: an admin's batch restore (AdminRelevanceService.restore) runs in its request; this finishes what a hide chunk
    // that outran it, or a request that broke off, left: live automatic and admin holds of a batch whose case an admin
    // restored and that has no current case. Same chunks and settings as the hide; stops at once if a case becomes current.
    // shortcut: each tick reads every closed RESTORED_BY_ADMIN case (no index on close_reason) and probes the hold index
    // of each one's batch; add a partial index on relevance_cases WHERE close_reason = 'RESTORED_BY_ADMIN' by migration
    // once admin restores reach the tens of thousands.
    private fun finishAdminRestores(now: OffsetDateTime) {
        for (batchId in cases.findAdminRestoresLeft()) {
            try {
                for (i in 1..props.hide.chunksPerTick) {
                    val n = transactions.execute {
                        if (cases.currentFor(batchId) != null) 0 else restore(holds.restoreByAdminChunk(batchId, now, props.hide.chunkSize))
                    } ?: 0
                    if (n == 0) break
                }
            } catch (ex: Exception) {
                log.warn("Relevance admin restore step failed batch={} (retried next tick): {}", batchId, ex.message)
            }
        }
    }

    private fun hideBatch(target: HoldTarget, now: OffsetDateTime) {
        val done = inChunks(target) {
            hide(holds.hideChunk(target.batchId, target.caseId, target.eventId, target.photographerId, AUTO, now, props.hide.chunkSize))
        }
        if (done && !target.hideFinished) finishHide(target, now)
    }

    // Up to chunks-per-tick chunks, each its own short transaction that touches only hold rows and photos, while the case
    // is still the current target (a plain read: it may have been superseded since the list was read). True when a
    // chunk found nothing left.
    private fun inChunks(target: HoldTarget, chunk: () -> Int): Boolean {
        repeat(props.hide.chunksPerTick) {
            val n = transactions.execute { if (cases.isHoldTarget(target.caseId, target.action)) chunk() else -1 } ?: -1
            if (n < 0) return false
            if (n == 0) return true
        }
        return false
    }

    // The holds were just written: lock their photos in id order, then one conditional update.
    private fun hide(ids: List<UUID>): Int {
        if (ids.isNotEmpty()) {
            photos.lockIdsForUpdate(ids)
            holds.hidePhotos(ids)
        }
        return ids.size
    }

    // The holds were just stamped restored: lock their photos in id order, then un-hide by the one restore rule.
    private fun restore(ids: List<UUID>): Int {
        if (ids.isNotEmpty()) {
            photos.lockIdsForUpdate(ids)
            holds.unhideReleased(ids)
        }
        return ids.size
    }

    // The first pass that finds nothing left: stamp the case once, log one automatic decision and raise the big-hide
    // alert at the real count. A separate short transaction, after the photo chunks (lock order, part 1 section 2).
    // skipEmpty: a junk-only hide of no photos writes nothing (it is checked again next tick, one count).
    private fun finishHide(target: HoldTarget, now: OffsetDateTime, skipEmpty: Boolean = false) {
        val raised = transactions.execute {
            val hidden = holds.countLiveHolds(target.batchId, AUTO).toInt()
            if (skipEmpty && hidden == 0) return@execute false
            if (cases.finishHide(target.caseId, hidden, now) != 1) return@execute false
            decisionLog.logAutomaticUserDecision(target.photographerId, "batch_auto_hidden",
                meta = mapOf("caseId" to target.caseId.toString(), "eventId" to target.eventId.toString(), "photos" to hidden))
            log.info("Relevance hide finished case={} photos={}", target.caseId, hidden)
            hidden >= props.alerts.bigHidePhotos &&
                raiseAlert(target.batchId, target.caseId, AlertKind.BIG_HIDE, hidden, wouldHave = false, now)
        }
        if (raised == true) registry.counter("qp.relevance.alerts", "kind", AlertKind.BIG_HIDE.name.lowercase()).increment()
    }

    // One stored admin alert, kept from doubling by its key; the live ping goes out after the commit.
    private fun raiseAlert(batchId: UUID, caseId: UUID, kind: AlertKind, photos: Int, wouldHave: Boolean, now: OffsetDateTime): Boolean {
        val alertId = UUID.randomUUID()
        val stored = alerts.raise(alertId, batchId, caseId, kind.name, "${kind.name.lowercase()}:$caseId", wouldHave, photos, now) == 1
        // Published inside the transaction: NotificationBroadcaster sends it after the commit, and drops it on a rollback.
        if (stored) publisher.publishEvent(AdminInboxEvent(payload = mapOf("type" to "relevance_alert",
            "entityId" to alertId.toString(), "actorId" to caseId.toString(), "occurredAt" to now.toString())))
        return stored
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
        // The automatic hold kind in flag_hidden_photos.held_by (V106).
        const val AUTO = "AUTO"
        const val MODEL_COLUMN = 60
        // Verdict notes of a check that got no usable answer (plan 5.11).
        val FAILED_CALL = setOf("CALL_FAILED", "CUT_OFF", "BAD_JSON", "UNKNOWN_VALUE")
        // Verdicts whose auto-hide case gets the full scan (A.3 step 5); a whole-batch hide is not scanned.
        val SCANNED = setOf(CaseVerdict.FITS, CaseVerdict.MIXED)
        // Whole-batch findings a re-check never lifts (G7).
        val NEVER_LIFTED = setOf(CaseVerdict.WRONG_EVENT.name, CaseVerdict.COPIED.name)
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
