package com.quickpitik.service.relevance

import com.quickpitik.common.ErrorCodes
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
        notifier.deliver(photographerId, caseId, "relevance:${appeal.id}:filed",
            RelevanceNotices.appealReceived(row.eventName, copied = row.verdict == CaseVerdict.COPIED.name))
        log.info("Relevance appeal filed case={} appeal={}", caseId, appeal.id)
        return view(row, appealView(appeal, now))
    }

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
    )

    private fun appealView(appeal: RelevanceAppeal, createdAt: OffsetDateTime? = appeal.createdAt) =
        PhotographerAppealDto(appeal.id, appeal.status, createdAt, appeal.decidedAt)

    private fun notFound() = NotFoundException("Case not found", ErrorCodes.NOT_FOUND)

    private fun nothingToAppeal() = ConflictException("This case has nothing to appeal", ErrorCodes.NOTHING_TO_APPEAL)

    private companion object {
        const val AUTO = "AUTO_HIDE"
        const val MAX_REASON = 1000
        val HIDING = setOf(CaseAction.HIDE_JUNK.name, CaseAction.HIDE_ALL.name)
    }
}
