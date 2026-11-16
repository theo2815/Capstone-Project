package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.CaseVerdict
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.repository.OwedNote
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.events.postgresTimestamp
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

// RV-3.1 (part 2, A.5): the notify step of the check tick, database only. Each note an auto-hide case owes is one inbox
// message in approved words (RelevanceNotices) with its case id, stored once under its notice key
// (relevance:<caseId>:verdict, :blurry or :restored), then e-mailed after commit (RelevanceNoticeEmailListener). A verdict
// note stamps notice_at, which starts the appeal clock. Record-only tells no one (P18).
@Service
class RelevanceNotifier(
    private val cases: RelevanceCaseRepository,
    private val decisionLog: AdminDecisionLogService,
    private val switch: RelevanceSwitch,
    private val props: RelevanceProperties,
    private val clock: Clock,
    private val transactions: TransactionTemplate,
    private val publisher: ApplicationEventPublisher,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // Not notify(): that name is Object.notify.
    fun notifyOwed() {
        if (switch.mode != RelevanceMode.AUTO_HIDE) return
        val now = clock.postgresTimestamp()
        for (row in cases.findOwedNotes(props.check.maxBatchesPerTick)) {
            try {
                send(row, now)
            } catch (ex: Exception) {
                // A second server copy that stored the same key first lands here too (unique notice_key).
                log.warn("Relevance note step failed case={} note={} (retried next tick): {}", row.caseId, row.note, ex.message)
            }
        }
    }

    // One short transaction: the clock stamp (verdict notes only) and the message, so neither stands without the other.
    private fun send(row: OwedNote, now: OffsetDateTime) {
        transactions.executeWithoutResult {
            val note = noteFor(row, appealBy(now)) ?: return@executeWithoutResult
            if (row.note == VERDICT && cases.stampNotice(row.caseId, now) != 1) return@executeWithoutResult
            deliver(row.photographerId, row.caseId, "relevance:${row.caseId}:${row.note}", note)
            log.info("Relevance note stored case={} note={}", row.caseId, row.note)
        }
    }

    // One note in the caller's transaction: stored once under its key, then e-mailed after commit. RV-3.2a's appeal
    // notes (A.5 keys relevance:<appealId>:filed and so on) go out here too.
    fun deliver(photographerId: UUID, caseId: UUID, noticeKey: String, note: RelevanceNote) {
        val message = decisionLog.pushMessage(photographerId, PhotographerMessageKind.RELEVANCE_NOTICE, note.body,
            title = note.title, noticeKey = noticeKey, relevanceCaseId = caseId)
        publisher.publishEvent(RelevanceNoticeStored(message.id))
    }

    // {appealBy}: notice_at plus the appeal window, as a date in Philippine time.
    internal fun appealBy(noticeAt: OffsetDateTime): String = RelevanceNotices.date(appealLastDay(noticeAt, props.appeal.windowDays))

    internal fun noteFor(row: OwedNote, appealBy: String): RelevanceNote? {
        val sample = row.measured ?: 0
        val junk = row.junk ?: 0
        return when (row.note) {
            // RV-4.5a: an admin's hide of a no-consent case (the check did reach a finding, so not V2-T23).
            VERDICT -> if (row.verdictNote == ADMIN_HIDE && row.verdict != CaseVerdict.CANT_TELL.name) {
                RelevanceNotices.adminHideAfterCheck(row.eventName, appealBy)
            } else when (row.verdict?.let { CaseVerdict.valueOf(it) }) {
                CaseVerdict.MIXED -> RelevanceNotices.mixed(row.eventName, sample, row.batchPhotos, junk, appealBy)
                CaseVerdict.JUNK_DUMP -> RelevanceNotices.junkDump(row.eventName, sample, row.batchPhotos, junk, appealBy)
                CaseVerdict.WRONG_EVENT -> RelevanceNotices.wrongEvent(row.eventName, row.batchPhotos,
                    RelevanceNotices.date(LocalDate.parse(row.eventDate)), appealBy)
                CaseVerdict.COPIED -> RelevanceNotices.copied(row.eventName, row.batchPhotos, appealBy)
                // A whole-batch Can't tell case is only ever an admin's hide (RV-2.5b).
                CaseVerdict.CANT_TELL -> RelevanceNotices.adminHide(row.eventName, appealBy)
                CaseVerdict.FITS, null -> null
            }
            BLURRY -> RelevanceNotices.blurry(row.eventName, row.scanMeasured ?: 0, row.scanBlurry ?: 0)
            else -> if (row.closeReason == "RECHECK_LIFTED") RelevanceNotices.restoredAfterRecheck(row.eventName)
                else RelevanceNotices.restoredAfterScan(row.eventName)
        }
    }

    private companion object {
        const val VERDICT = "verdict"
        const val BLURRY = "blurry"
        const val ADMIN_HIDE = "ADMIN_HIDE"
    }
}

// The last day to appeal: the Philippine date of notice_at plus the appeal window. The note names it (V2-T11 to V2-T14,
// V2-T23) and filing is open through the end of it (RV-3.2a).
fun appealLastDay(noticeAt: OffsetDateTime, windowDays: Long): LocalDate =
    noticeAt.atZoneSameInstant(EventDtoMapper.PH_ZONE).toLocalDate().plusDays(windowDays)
