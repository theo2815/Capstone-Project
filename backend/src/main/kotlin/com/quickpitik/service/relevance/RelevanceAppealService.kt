package com.quickpitik.service.relevance

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.RelevanceAppealRowDto
import com.quickpitik.dto.admin.toDto
import com.quickpitik.exception.ApiException
import com.quickpitik.service.admin.AdminDecisionLogService
import org.springframework.http.HttpStatus
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.dto.photographer.PhotographerAppealDto
import com.quickpitik.dto.photographer.PhotographerRelevanceCaseDto
import com.quickpitik.entity.AlertKind
import com.quickpitik.entity.AppealStatus
import com.quickpitik.entity.CaseAction
import com.quickpitik.entity.CaseStage
import com.quickpitik.entity.CaseVerdict
import com.quickpitik.entity.RelevanceAppeal
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.OwnCaseRow
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceAppealRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.events.postgresTimestamp
import com.quickpitik.websocket.AdminInboxEvent
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.Clock
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID

// RV-3.2a (part 2, A.6, G11): a photographer reads their own case and appeals it. An appeal is open to the current
// auto-hide case of the photographer's own batch when it hides something (target HIDE_JUNK or HIDE_ALL, an admin's hide
// of a Can't tell batch included) and its note started the clock (notice_at), through the last day the note names,
// once per case, with no other appeal open on the batch. Filing stores the reason, raises one APPEAL alert, pings
// admins and sends V2-T17. Deciding is RV-3.2b. With the mode not auto-hide, nothing runs and nothing can be appealed.
@Service
class RelevanceAppealService(
    private val cases: RelevanceCaseRepository,
    private val appeals: RelevanceAppealRepository,
    private val alerts: RelevanceAlertRepository,
    private val notifier: RelevanceNotifier,
    private val switch: RelevanceSwitch,
    private val props: RelevanceProperties,
    private val clock: Clock,
    private val publisher: ApplicationEventPublisher,
    private val confirmer: RelevanceConfirmer,
    private val decisionLog: AdminDecisionLogService,
    private val payoutHold: RelevanceHoldService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // Own auto-hide cases only, whatever their stage; anything else is not found.
    @Transactional(readOnly = true)
    fun caseFor(photographerId: UUID, caseId: UUID): PhotographerRelevanceCaseDto {
        if (switch.mode != RelevanceMode.AUTO_HIDE) throw notFound()
        val row = cases.ownCase(caseId, photographerId)?.takeIf { it.appliedMode == AUTO } ?: throw notFound()
        return view(row, appeals.findByCaseId(caseId)?.let { appealView(it) })
    }

    @Transactional
    fun file(photographerId: UUID, caseId: UUID, reason: String): PhotographerRelevanceCaseDto {
        if (switch.mode != RelevanceMode.AUTO_HIDE) throw nothingToAppeal()
        val text = reason.trim()
        // Postgres refuses a NUL character in TEXT: a 400 here, not a 500 at the insert.
        if (text.isEmpty() || text.length > MAX_REASON || '\u0000' in text) {
            throw ValidationException("An appeal reason must be 1 to 1,000 characters", field = "reason")
        }
        cases.lockCase(caseId) ?: throw notFound()
        val row = cases.ownCase(caseId, photographerId) ?: throw notFound()
        if (appeals.findByCaseId(caseId) != null) {
            // The three refusal messages are V2-T22's approved words (the website shows the server's message).
            throw ConflictException("You have already appealed this decision.", ErrorCodes.APPEAL_ALREADY_FILED)
        }
        val noticeAt = row.noticeAt?.atOffset(ZoneOffset.UTC)
        if (row.stage != CaseStage.APPLIED.name || row.appliedMode != AUTO || row.action !in HIDING || noticeAt == null) {
            throw nothingToAppeal()
        }
        val now = clock.postgresTimestamp()
        if (now.atZoneSameInstant(EventDtoMapper.PH_ZONE).toLocalDate().isAfter(appealLastDay(noticeAt, props.appeal.windowDays))) {
            throw ConflictException("The time to appeal this decision has ended.", ErrorCodes.APPEAL_WINDOW_CLOSED)
        }
        if (appeals.existsByBatchIdAndStatus(row.batchId, AppealStatus.OPEN.name)) {
            throw ConflictException("You already have an open appeal for your photos in this event. " +
                "We will tell you in your inbox when it is decided.", ErrorCodes.APPEAL_ALREADY_OPEN)
        }
        val appeal = appeals.saveAndFlush(RelevanceAppeal(caseId = caseId, batchId = row.batchId, photographerId = photographerId, reason = text))
        raiseAlert(row, appeal.id, now)
        // RV-4.5a review: the payouts sentence only while a payout hold binds the case.
        notifier.deliver(photographerId, caseId, "relevance:${appeal.id}:filed",
            RelevanceNotices.appealReceived(row.eventName, copied = cases.liveHoldFlag(caseId) != null))
        log.info("Relevance appeal filed case={} appeal={}", caseId, appeal.id)
        return view(row, appealView(appeal, now))
    }

    // RV-3.2 part b: the admin's appeals list, in any mode (like the page's other reads).
    @Transactional(readOnly = true)
    fun listAppeals(status: String, page: PaginationParams): PaginatedResponse<RelevanceAppealRowDto> {
        if (status !in LIST_STATUSES) throw ValidationException("status must be one of ${LIST_STATUSES.joinToString()}", field = "status")
        val result = appeals.pageAppeals(status, OffsetLimitPageable(page))
        return PaginatedResponse.of(result.content.map { it.toDto() }, result.totalElements, page)
    }

    // A.6 (addendum line 46): the page calls this when it shows the samples. The latest admin to open it is the one who
    // may decide. In auto-hide only, as the admin's other acting routes.
    @Transactional
    fun evidenceOpened(adminId: UUID, appealId: UUID): RelevanceAppealRowDto {
        actionsOn()
        val appeal = appeals.lockAppeal(appealId) ?: throw appealNotFound()
        if (appeal.status != AppealStatus.OPEN.name) throw alreadyDecided()
        appeal.evidenceOpenedAt = clock.postgresTimestamp()
        appeal.evidenceOpenedBy = adminId
        appeals.save(appeal)
        return appeals.appealRow(appealId)!!.toDto()
    }

    // A.6: the admin decides an open appeal whose evidence they opened. Won: the case's target is lowered (winAppeal) and
    // V2-T18 goes out; closing a won whole-batch case APPEAL_WON when its scan has brought the good photos back is part c.
    // Lost: a whole-batch finding is confirmed at once (A.7) and V2-T19 goes out. The note stays with admins (G12).
    @Transactional
    fun decide(adminId: UUID, appealId: UUID, outcome: String, note: String?): RelevanceAppealRowDto {
        actionsOn()
        val result = listOf(AppealStatus.WON, AppealStatus.LOST).firstOrNull { it.name == outcome }
            ?: throw ValidationException("outcome must be WON or LOST", field = "outcome")
        // Postgres refuses a NUL character in TEXT: a 400 here, not a 500 at the update.
        if (note != null && '\u0000' in note) throw ValidationException("The note cannot contain a NUL character", field = "note")
        // A plain row, not the entity: loading the entity before the lock would hand back a stale copy from the session.
        val caseId = appeals.appealRow(appealId)?.caseId ?: throw appealNotFound()
        // The case first, then the appeal: the same order as filing.
        cases.lockCase(caseId)
        val appeal = appeals.lockAppeal(appealId) ?: throw appealNotFound()
        if (appeal.status != AppealStatus.OPEN.name) throw alreadyDecided()
        if (appeal.evidenceOpenedBy != adminId) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.EVIDENCE_NOT_OPENED, "Open this appeal's evidence before deciding it")
        }
        val now = clock.postgresTimestamp()
        appeal.status = result.name
        appeal.decidedAt = now
        appeal.decidedBy = adminId
        appeal.decisionNote = note?.trim()?.ifBlank { null }
        appeals.save(appeal)
        val case = cases.findById(caseId).orElseThrow()
        val acting = case.appliedMode == AUTO && case.stage == CaseStage.APPLIED.name
        val wholeBatch = case.action == CaseAction.HIDE_ALL.name
        if (acting) {
            if (result == AppealStatus.WON) {
                cases.winAppeal(caseId)
                // RV-4.2 (A.6, A.8): a won appeal ends a copied case's payout hold.
                payoutHold.end(caseId, adminId, restore = true)
            } else confirmer.confirm(caseId, now)
        }
        decisionLog.logUserDecision(adminId, appeal.photographerId, "relevance_appeal_${result.name.lowercase()}", appeal.decisionNote,
            meta = mapOf("caseId" to caseId.toString(), "appealId" to appealId.toString()))
        // Only a decision that acts on the case is told: for a case closed meanwhile (an admin restored the batch) the
        // notes would describe a hide that is gone. The appeal is still decided, so it never blocks a later one.
        if (acting) {
            val eventName = cases.ownCase(caseId, appeal.photographerId)!!.eventName
            val text = if (result == AppealStatus.WON) RelevanceNotices.appealWon(eventName, wholeBatch)
                // RV-4.5a review: the payouts sentence only while a hold binds; an admin's hide never counts.
                else RelevanceNotices.appealLost(eventName, copied = cases.liveHoldFlag(caseId) != null,
                    counts = case.verdictNote != "ADMIN_HIDE")
            notifier.deliver(appeal.photographerId, caseId, "relevance:$appealId:${result.name.lowercase()}", text)
        }
        log.info("Relevance appeal decided appeal={} case={} outcome={}", appealId, caseId, result)
        return appeals.appealRow(appealId)!!.toDto()
    }

    private fun actionsOn() {
        if (switch.mode != RelevanceMode.AUTO_HIDE) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.RELEVANCE_ACTIONS_OFF, "Deciding appeals needs the relevance mode auto-hide")
        }
    }

    private fun appealNotFound() = NotFoundException("Relevance appeal not found", ErrorCodes.RELEVANCE_APPEAL_NOT_FOUND)

    private fun alreadyDecided() = ApiException(HttpStatus.CONFLICT, ErrorCodes.APPEAL_ALREADY_DECIDED, "This appeal has already been decided")

    // One stored APPEAL alert per appeal; the live ping goes out after the commit (NotificationBroadcaster).
    private fun raiseAlert(row: OwnCaseRow, appealId: UUID, now: OffsetDateTime) {
        val alertId = UUID.randomUUID()
        if (alerts.raise(alertId, row.batchId, row.caseId, AlertKind.APPEAL.name, "appeal:$appealId", false, row.hiddenPhotos, now) == 1) {
            publisher.publishEvent(AdminInboxEvent(payload = mapOf("type" to "relevance_alert",
                "entityId" to alertId.toString(), "actorId" to row.caseId.toString(), "occurredAt" to now.toString())))
        }
    }

    private fun view(row: OwnCaseRow, appeal: PhotographerAppealDto?) = PhotographerRelevanceCaseDto(
        caseId = row.caseId, eventId = row.eventId, eventName = row.eventName, verdict = row.verdict, action = row.action,
        current = row.stage == CaseStage.APPLIED.name, batchPhotos = row.batchPhotos, sampleSize = row.measured,
        junkCount = row.junk, hiddenPhotos = row.hiddenPhotos, scanMeasured = row.scanMeasured, scanBlurry = row.scanBlurry,
        noticeAt = row.noticeAt,
        appealDeadline = row.noticeAt?.let { appealLastDay(it.atOffset(ZoneOffset.UTC), props.appeal.windowDays) },
        appeal = appeal,
        eventDate = row.eventDate?.let { LocalDate.parse(it) },
        closeReason = row.closeReason,
    )

    private fun appealView(appeal: RelevanceAppeal, createdAt: OffsetDateTime? = appeal.createdAt) =
        PhotographerAppealDto(appeal.id, appeal.status, createdAt, appeal.decidedAt)

    private fun notFound() = NotFoundException("Case not found", ErrorCodes.NOT_FOUND)

    private fun nothingToAppeal() = ConflictException("This case has nothing to appeal", ErrorCodes.NOTHING_TO_APPEAL)

    private companion object {
        const val AUTO = "AUTO_HIDE"
        const val MAX_REASON = 1000
        val HIDING = setOf(CaseAction.HIDE_JUNK.name, CaseAction.HIDE_ALL.name)
        val LIST_STATUSES = listOf("OPEN", "WON", "LOST", "all")
    }
}
