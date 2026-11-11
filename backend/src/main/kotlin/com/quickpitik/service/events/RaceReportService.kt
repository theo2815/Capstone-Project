package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.RaceProperties
import com.quickpitik.dto.report.ReportResultDto
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.Role
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.SystemFlagService
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.Clock
import java.util.UUID

// AA-R9: signed-in users report an event or a photographer; each report is a
// user_report flag in the admin queue (one open report per account per target).
// Enough distinct accounts hide an unproven typed race (Net 3); a proven race,
// an untyped event or a photographer is only ever flagged. Only an admin rejects.
//
// shortcut: two reports landing at the same moment can each count one short and
// neither hides; the next report (or the proof sweep) catches it.
@Service
class RaceReportService(
    private val eventRepository: EventRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
    private val flagRepository: FlagRepository,
    private val flags: SystemFlagService,
    private val raceProof: RaceProofService,
    private val props: RaceProperties,
    private val clock: Clock,
) {

    @Transactional
    fun reportEvent(reporterId: UUID, eventId: UUID, note: String): ReportResultDto {
        val event = eventRepository.findById(eventId).orElse(null)
            ?.takeIf { it.deletedAt == null && it.status != EventStatus.DRAFT }
            ?: throw NotFoundException("Event not found", ErrorCodes.EVENT_NOT_FOUND)
        val raised = flags.raise(FlagTargetKind.EVENT, event.id, SystemFlagReason.USER_REPORT, note.trim(), reporterId)
        if (raised && event.eventType != null && event.hiddenAt == null) {
            val reporters = flagRepository.countDistinctReporters(FlagTargetKind.EVENT.wire, event.id)
            if (reporters >= props.reports.hideThreshold && !raceProof.isProven(event) &&
                eventRepository.hideUnprovenRace(event.id, clock.postgresTimestamp()) == 1
            ) {
                flags.raise(
                    FlagTargetKind.EVENT, event.id, SystemFlagReason.REPORT_THRESHOLD_HIDDEN,
                    note = "Hidden after reports from $reporters accounts",
                )
            }
        }
        return ReportResultDto(alreadyReported = !raised)
    }

    @Transactional
    fun reportPhotographer(reporterId: UUID, handle: String, note: String): ReportResultDto {
        val photographer = photographerSettingsRepository.findByHandleIgnoreCase(handle.trim().lowercase())
            ?.let { userRepository.findById(it.userId).orElse(null) }
            ?.takeIf { it.role == Role.PHOTOGRAPHER }
            ?: throw NotFoundException("Photographer not found", ErrorCodes.USER_NOT_FOUND)
        val raised = flags.raise(FlagTargetKind.USER, photographer.id, SystemFlagReason.USER_REPORT, note.trim(), reporterId)
        return ReportResultDto(alreadyReported = !raised)
    }
}
