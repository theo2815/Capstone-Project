package com.quickpitik.service.events

import com.quickpitik.config.RaceProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RaceProofCounts
import com.quickpitik.service.admin.SystemFlagService
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId

// AA-R8 (Net 2): when a typed race's uploads close (race day + 3, Manila), a race
// nobody proved leaves /events and the pickers, and admins get a flag. Nothing is
// deleted and only an admin rejects; an admin approval or relist always counts as
// proof. Races sharing many bib numbers with another live event are flagged too.
// Each race is claimed once (proof_checked_at), so a second run does nothing.
@Service
class RaceProofService(
    private val eventRepository: EventRepository,
    private val photoRepository: PhotoRepository,
    private val flags: SystemFlagService,
    private val props: RaceProperties,
    private val clock: Clock,
    private val transactionTemplate: TransactionTemplate,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun isProven(event: Event): Boolean = proven(event, photoRepository.raceProofCounts(event.id))

    // Not gated on AI (two photographers prove a race without it), so the first run waits one
    // interval: a boot (each deploy, each test context) never sweeps at once.
    @Scheduled(
        initialDelayString = "\${app.races.proof.sweep-interval-ms:3600000}",
        fixedDelayString = "\${app.races.proof.sweep-interval-ms:3600000}",
    )
    fun sweep() {
        val closedOnOrBefore = LocalDate.now(clock.withZone(PH_ZONE)).minusDays(UPLOAD_WINDOW_DAYS + 1)
        eventRepository.findRacesDueForProof(closedOnOrBefore, PageRequest.of(0, BATCH)).forEach { race ->
            try {
                transactionTemplate.execute { check(race) }
            } catch (ex: Exception) {
                log.warn("Race proof check failed for race {} (retried next run): {}", race.id, ex.message)
            }
        }
    }

    private fun check(race: Event) {
        val now = clock.postgresTimestamp()
        if (eventRepository.claimProofCheck(race.id, now) == 0) return
        flagSharedBibs(race, now)
        val c = photoRepository.raceProofCounts(race.id)
        val p = props.proof
        val outcome = when {
            proven(race, c) -> "proven"
            c.total > 0 && c.indexed.toDouble() / c.total < p.minIndexedShare ->
                flag(race, SystemFlagReason.NO_PROOF_UNINDEXED, c, "flagged, mostly unindexed")
            // The photographer rule failed; with enough indexed photos only the bib rule did.
            race.eventType in p.bibFlagOnlyTypes && c.indexed >= p.minIndexedPhotos ->
                flag(race, SystemFlagReason.NO_PROOF_UNINDEXED, c, "flagged, bib rule on a flag-only type")
            eventRepository.hideUnprovenRace(race.id, now) == 0 -> "kept, already hidden or admin-approved"
            else -> flag(race, SystemFlagReason.NO_PROOF_HIDDEN, c, "hidden")
        }
        log.info("Race proof: race {} {} ({})", race.id, outcome, summary(c))
    }

    private fun proven(event: Event, c: RaceProofCounts): Boolean {
        val p = props.proof
        return event.adminApprovedAt != null ||
            c.photographers >= p.minPhotographers ||
            (c.indexed >= p.minIndexedPhotos && c.withBib.toDouble() / c.indexed >= p.minBibShare)
    }

    private fun flag(race: Event, reason: SystemFlagReason, c: RaceProofCounts, outcome: String): String {
        flags.raise(FlagTargetKind.EVENT, race.id, reason, note = "No proof when uploads closed: ${summary(c)}")
        return outcome
    }

    private fun flagSharedBibs(race: Event, now: OffsetDateTime) {
        // Same date tolerance as the duplicate matcher: one race, maybe entered a day off.
        val days = props.duplicate.dateToleranceDays
        val matches = photoRepository.eventsSharingBibs(
            race.id, race.date.minusDays(days), race.date.plusDays(days), props.sameBib.minSharedBibs,
        )
        if (matches.isEmpty()) return
        val note = matches.joinToString("\n") { "${it.name} (${it.eventId}): ${it.shared} bibs" }
        if (!flags.raise(FlagTargetKind.EVENT, race.id, SystemFlagReason.SAME_BIB_DUPLICATE, note = note)) {
            flags.appendNote(FlagTargetKind.EVENT, race.id, SystemFlagReason.SAME_BIB_DUPLICATE, "$now $note")
        }
    }

    private fun summary(c: RaceProofCounts) =
        "photographers=${c.photographers} photos=${c.total} indexed=${c.indexed} withBib=${c.withBib}"

    private companion object {
        val PH_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        const val UPLOAD_WINDOW_DAYS = 3L // matches PhotoUploadService [date, date+3d]
        // Claimed races drop out of the query, so the next run takes the next page.
        const val BATCH = 100
    }
}
