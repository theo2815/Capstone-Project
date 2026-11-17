package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.AlertKind
import com.quickpitik.websocket.AdminInboxEvent
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceStrikeRepository
import org.springframework.context.ApplicationEventPublisher
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

// RV-3.2 part b (A.7): confirming a case. It stamps confirmed_at and writes the case's one relevance_strikes row, in the
// caller's transaction; true only for the call that confirmed it, so a second call changes nothing. A lost appeal and
// RV-4.1's confirm step (RelevanceCheckService.confirmDue) both call it, so both climb the same ladder.
@Service
class RelevanceConfirmer(
    private val cases: RelevanceCaseRepository,
    private val strikes: RelevanceStrikeRepository,
    private val alerts: RelevanceAlertRepository,
    private val events: EventRepository,
    private val notifier: RelevanceNotifier,
    private val props: RelevanceProperties,
    private val publisher: ApplicationEventPublisher,
) {
    // Lock order: the case, then the photographer's user row (a lost appeal holds the case, then the appeal, already).
    @Transactional
    fun confirm(caseId: UUID, now: OffsetDateTime): Boolean {
        // Its own statement: once a filing that holds the case commits, the stamp's fresh snapshot sees the open appeal.
        cases.lockCase(caseId) ?: return false
        if (cases.stampConfirmed(caseId, now) != 1) return false
        strikes.lockPhotographerOf(caseId)
        strikes.insertForCase(caseId)
        climb(caseId, now)
        return true
    }

    // RV-4.1 (A.7, G6): the photographer's strikes in the rolling window, across events, this one included. The second
    // sends one V2-T20 note; the third and each later case raise one THIRD_CASE alert of their own (key per case). An
    // admin decides on suspension (AdminUserService.suspend): nothing here suspends, fines or marks anyone.
    private fun climb(caseId: UUID, now: OffsetDateTime) {
        val strike = strikes.findByCaseId(caseId) ?: return
        val months = props.strikes.windowMonths
        val count = strikes.countSince(strike.photographerId, now.minusMonths(months.toLong()))
        if (count == 2) {
            val eventName = events.findById(strike.eventId).orElseThrow().name
            notifier.deliver(strike.photographerId, caseId, "relevance:strike2:$caseId", RelevanceNotices.secondCase(eventName, months))
        } else if (count >= 3) {
            val case = cases.findById(caseId).orElseThrow()
            val alertId = UUID.randomUUID()
            if (alerts.raise(alertId, case.batchId, caseId, AlertKind.THIRD_CASE.name, "third_case:$caseId", false, case.hiddenPhotos, now) == 1) {
                // NotificationBroadcaster sends the ping after the commit, and drops it on a rollback.
                publisher.publishEvent(AdminInboxEvent(payload = mapOf("type" to "relevance_alert",
                    "entityId" to alertId.toString(), "actorId" to caseId.toString(), "occurredAt" to now.toString())))
            }
        }
    }
}
