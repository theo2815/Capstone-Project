package com.quickpitik.service.relevance

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.entity.BatchRelevance
import com.quickpitik.entity.FlagStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.DueBatch
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.events.postgresTimestamp
import io.micrometer.core.instrument.MeterRegistry
import io.micrometer.core.instrument.Timer
import org.slf4j.LoggerFactory
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.concurrent.TimeUnit

// Database evidence only: this sweep never calls a provider or changes a photo or payout. In flag mode
// it also opens or extends one admin flag per photographer and may send the wrong-event inbox note.
@Service
class BatchRelevanceService(
    private val repository: BatchRelevanceRepository,
    private val props: RelevanceProperties,
    private val clock: Clock,
    private val transactions: TransactionTemplate,
    private val registry: MeterRegistry,
    private val systemFlagService: SystemFlagService,
    private val flagRepository: FlagRepository,
    private val userRepository: UserRepository,
    private val adminProperties: AdminProperties,
    private val adminDecisionLogService: AdminDecisionLogService,
    private val eventRepository: EventRepository,
    // Spring injects the bean; the default lets the existing tests build the service from props alone.
    private val switch: RelevanceSwitch = RelevanceSwitch(props, adminProperties, AiApiProperties()),
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // P9: with the admin flags switch off nobody could see or close a flag, so flag mode acts as shadow.
    private val raising = switch.v1Flagging

    @Scheduled(
        initialDelayString = "\${app.relevance.sweep-interval-ms:60000}",
        fixedDelayString = "\${app.relevance.sweep-interval-ms:60000}",
    )
    fun sweep() {
        if (switch.mode == RelevanceMode.OFF) return
        val now = clock.postgresTimestamp()
        // RV-1.2 (G14): before the due list, because a settled batch is never due again.
        try {
            transactions.execute {
                repository.purgeRareWords(props.words.minSharePercent, now.minusDays(props.settleDays), props.maxBatchesPerTick, now)
            }
        } catch (ex: Exception) {
            log.warn("Relevance word purge failed (retried next tick): {}", ex.javaClass.simpleName)
        }
        val today = LocalDate.now(clock.withZone(EventDtoMapper.PH_ZONE))
        val cutoffs = uploadCutoffs(today)
        val due = repository.findDueBatches(now.minusMinutes(props.quietMinutes), now.minusMinutes(props.rescoreMinutes),
            cutoffs.adminClosedOnOrBefore, cutoffs.ownerClosedOnOrBefore, props.maxBatchesPerTick, raising)
        if (due.isEmpty()) return
        val timer = Timer.start(registry)
        var budget = props.lookupsPerTick
        var worked = 0
        var scored = 0
        var low = 0
        for (batch in due) {
            try {
                val row = transactions.execute {
                    repository.ensureBatch(batch.eventId, batch.photographerId)
                    repository.findByEventIdAndPhotographerId(batch.eventId, batch.photographerId)
                } ?: continue
                val pending = repository.countPendingLookups(batch.eventId, batch.photographerId) > 0
                if (pending && budget == 0) continue
                transactions.execute { repository.touchSwept(row.id, now) }
                worked++
                if (pending) budget -= lookUp(batch, budget)
                if (repository.countPendingLookups(batch.eventId, batch.photographerId) == 0L) {
                    val result = score(batch, today) ?: continue
                    scored++
                    if (result.level == "LOW") low++
                }
            } catch (ex: Exception) {
                log.warn("Relevance batch failed event={} photographer={} (retried next tick): {}",
                    batch.eventId, batch.photographerId, ex.message)
            }
        }
        if (worked > 0) {
            val elapsed = timer.stop(registry.timer("qp.relevance.tick"))
            log.info("Relevance sweep batches={} lookups={} scored={} low={} ms={}",
                worked, props.lookupsPerTick - budget, scored, low, TimeUnit.NANOSECONDS.toMillis(elapsed))
        }
    }

    internal fun lookUp(batch: DueBatch, budget: Int): Int {
        val inserted = transactions.execute {
            repository.setLocalStatementTimeout(props.lookupTimeout.toMillis())
            val row = repository.findByEventIdAndPhotographerId(batch.eventId, batch.photographerId) ?: return@execute 0
            repository.lookUpNext(row.id, batch.eventId, batch.photographerId, minOf(budget, props.lookupsPerTick),
                props.copy.weakDistance, clock.postgresTimestamp())
        }!!
        registry.counter("qp.relevance.lookups").increment(inserted.toDouble())
        return inserted
    }

    internal fun score(batch: DueBatch, today: LocalDate): BatchRelevance? {
        // Look the id up outside: an entity loaded before lockById would keep stale fields in this transaction.
        val id = repository.findByEventIdAndPhotographerId(batch.eventId, batch.photographerId)?.id ?: return null
        val result = transactions.execute {
            val row = repository.lockById(id) ?: return@execute null
            val closed = lastUploadDate(batch.eventDate, batch.sharedRace)
            val settled = today > closed.plusDays(props.settleDays)
            val counts = repository.batchCounts(batch.eventId, batch.photographerId, props.copy.tightDistance, props.copy.weakDistance)
            row.photosLive = counts.photosLive.toInt()
            row.photosProcessing = counts.photosProcessing.toInt()
            row.indexedPhotos = counts.indexedPhotos.toInt()
            row.lookups = counts.lookups.toInt()
            row.tightHits = counts.tightHits.toInt()
            row.weakHits = counts.weakHits.toInt()
            row.exactMatches = counts.exactMatches.toInt()
            row.xmpMatches = repository.xmpMatches(batch.eventId, batch.photographerId).toInt()
            if (batch.sharedRace) {
                val metadata = repository.metadataRows(batch.eventId, batch.photographerId)
                val dates = metadata.filter { it.state == PhotoMetadataState.READY }.mapNotNull { it.fields }.mapNotNull { fields ->
                    fields.takenAtLocal?.let { manilaDate(it, fields.takenAtOffset) }
                }
                row.datedPhotos = dates.size
                row.datedOutside = dates.count { outsideWindow(it, batch.eventDate, props.date.windowDays) }
                row.metadataPending = metadata.count { it.state == PhotoMetadataState.PENDING || it.state == PhotoMetadataState.PROCESSING }
                row.dateLow = dateLow(row.datedPhotos!!, row.datedOutside!!, if (settled) 0 else row.metadataPending!!, props.date)
            } else {
                row.datedPhotos = null
                row.datedOutside = null
                row.metadataPending = null
                row.dateLow = null
            }
            val bibs = if (counts.indexedPhotos > 0) repository.bibOverlap(batch.eventId, batch.photographerId) else null
            row.bibbedPhotos = bibs?.bibbedPhotos?.toInt()
            row.bibOverlapPhotos = bibs?.bibOverlapPhotos?.toInt()
            row.copyLow = copyLow(counts.tightHits.toInt(), counts.lookups.toInt(), props.copy)
            row.level = if (row.copyLow == true || row.dateLow == true) "LOW" else null
            val now = clock.postgresTimestamp()
            row.scoredAt = now
            row.scoredUploadAt = batch.lastUploadAt.atOffset(ZoneOffset.UTC)
            row.scoredEventDate = batch.eventDate
            row.finalScoredAt = if (today > closed &&
                ((row.metadataPending ?: 0) == 0 && counts.photosProcessing == 0L || settled)) now else null
            if (raising && row.level == "LOW") flag(row, batch, now)
            repository.save(row)
        } ?: return null
        registry.counter("qp.relevance.scored", "level", result.level?.lowercase() ?: "none").increment()
        log.info(formatRelevanceBatchLine(result))
        return result
    }

    // Plan 5.9, inside the scoring transaction, which already holds the batch row. Lock order: batch row,
    // then user row, then flags. No Flag is read before the user lock: appendNote saves the whole row, so
    // an earlier read could write back a stale status or hold (R2-35).
    private fun flag(row: BatchRelevance, batch: DueBatch, now: OffsetDateTime) {
        userRepository.lockById(row.photographerId)
        // Once per open flag; after a close (resolved, dismissed or hidden) only for uploads made after it (P6).
        row.flagId?.let { flagRepository.findById(it).orElse(null) }?.let { linked ->
            val closedAt = linked.resolvedAt
            if (linked.status in OPEN_FLAG || closedAt == null || !batch.lastUploadAt.isAfter(closedAt.toInstant())) return
        }
        val wrongEventNote = row.dateLow == true &&
            repository.existsOtherDateClear(row.eventId, row.photographerId, batch.eventDate)
        val line = "event ${row.eventId} ${batch.eventDate}: copy ${row.tightHits}/${row.lookups}, " +
            "dated outside ${row.datedOutside ?: "-"}/${row.datedPhotos ?: "-"}, exact ${row.exactMatches}, xmp ${row.xmpMatches}" +
            if (row.dateLow == true && !wrongEventNote) "; no other batch of this event has a usable date: check the event's date first" else ""
        val reason = SystemFlagReason.BATCH_RELEVANCE_LOW
        if (!systemFlagService.raise(FlagTargetKind.USER, row.photographerId, reason, note = line)) {
            systemFlagService.appendNote(FlagTargetKind.USER, row.photographerId, reason, "$now $line")
        }
        row.flagId = flagRepository.findByTargetKindWireAndTargetIdAndReasonAndStatusWireIn(
            FlagTargetKind.USER.wire, row.photographerId, reason.wire, OPEN_FLAG.map { it.wire },
        ).first { it.reporterId == null }.id
        row.flaggedAt = now
        if (wrongEventNote) {
            // PE-L4 text T1 (approved 2026-11-12). Names no other photographer (D18, C6).
            val event = eventRepository.findById(row.eventId).orElseThrow()
            adminDecisionLogService.pushMessage(
                photographerId = row.photographerId,
                kind = PhotographerMessageKind.ADMIN_MESSAGE,
                body = "Many of the photos you uploaded to ${event.name} have camera dates far from the event's date " +
                    "(${batch.eventDate}). For now they stay on sale, and an admin will take a look. " +
                    "If your camera's clock was set wrong, or the photos belong to a different event, contact support.",
                title = "Check the event for some uploads",
            )
        }
    }

    private companion object {
        val OPEN_FLAG = listOf(FlagStatus.OPEN, FlagStatus.ESCALATED)
    }
}

internal fun formatRelevanceBatchLine(row: BatchRelevance): String =
    "Relevance batch event=${row.eventId} photographer=${row.photographerId} level=${row.level?.lowercase() ?: "none"} " +
        "copyLow=${row.copyLow} dateLow=${row.dateLow} lookups=${row.lookups} tight=${row.tightHits} weak=${row.weakHits} " +
        "exact=${row.exactMatches} xmp=${row.xmpMatches} dated=${row.datedPhotos} outside=${row.datedOutside} " +
        "pendingMeta=${row.metadataPending} bibbed=${row.bibbedPhotos} bibOverlap=${row.bibOverlapPhotos} final=${row.finalScoredAt != null}"
