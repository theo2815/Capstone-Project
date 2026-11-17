package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.IdentityProperties
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.RaceProperties
import com.quickpitik.dto.photographer.CreateRaceRequest
import com.quickpitik.dto.photographer.PhotographerEventDetailDto
import com.quickpitik.dto.photographer.RaceCandidateDto
import com.quickpitik.dto.photographer.RaceCreateResultDto
import com.quickpitik.dto.photographer.UpdateRaceRequest
import com.quickpitik.dto.photographer.detailDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.EventVisibility
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.RaceCheckReason
import com.quickpitik.entity.RaceType
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.service.admin.AdminEventService
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.moderation.ImageModerationService
import com.quickpitik.service.moderation.ImageModerationVerdict
import com.quickpitik.service.moderation.WordList
import com.quickpitik.service.moderation.WordListChecker
import com.quickpitik.service.photographer.PhotographerAccessGuard
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.websocket.AdminInboxEvent
import org.hibernate.exception.ConstraintViolationException
import org.springframework.context.ApplicationEventPublisher
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

/**
 * Shared race events (AA-R3, auto-approval Plan 2). A race is an event with `eventType` set and
 * `createdBy` NULL; `submittedBy` names the photographer who created it. Create runs
 * "pre-check, upload the cover, then lock → re-check → insert" so two photographers creating the
 * same race at once leave one row; join adds a coverage row (and maybe a name alias).
 */
@Service
class RaceEventService(
    private val eventRepository: EventRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val sellerDetailsRepository: SellerDetailsRepository,
    private val accessGuard: PhotographerAccessGuard,
    private val raceProperties: RaceProperties,
    private val identityProperties: IdentityProperties,
    private val platformProperties: PlatformProperties,
    private val regionsService: RegionsService,
    private val eventCoverService: EventCoverService,
    private val eventDtoMapper: EventDtoMapper,
    private val systemFlagService: SystemFlagService,
    private val wordListChecker: WordListChecker,
    private val transactionTemplate: TransactionTemplate,
    private val eventPublisher: ApplicationEventPublisher,
    private val imageModerationService: ImageModerationService,
    private val clock: Clock,
    private val eventFaceExpiry: EventFaceExpiry,
) {
    private val matcher = raceProperties.duplicate.let { DuplicateMatcher(it.strongScore, it.mediumScore, it.dateToleranceDays) }

    private class Draft(val input: RaceInput, val type: RaceType, val req: CreateRaceRequest)

    // No @Transactional: the pre-check and the cover upload run outside any transaction, and
    // NOT_SUPPORTED would bind one EntityManager (and its connection) for the whole call, so a
    // create waiting on the lock would hold two pool connections instead of one.
    fun create(photographerId: UUID, req: CreateRaceRequest, cover: AdminEventService.CoverUpload?): RaceCreateResultDto {
        requireSwitchOn()
        val settings = accessGuard.requireApproved(photographerId, "creating a race")
        // Create needs passed identity whatever IDENTITY_GATE_ENABLED says, so an early switch-on
        // cannot let an unchecked photographer put a live, listed race on /events.
        if (sellerDetailsRepository.findById(photographerId).orElse(null)?.status?.passesIdentity() != true) {
            throw sellerDetailsRequired()
        }
        val draft = validate(req)
        val declared = req.confirmDifferentFrom
        // AA-R4: the text and date checks, and the daily cap counted before moderation.
        val reasons = checksFor(draft.input, null, photographerId, req.description).toMutableSet()
        stopOutcome(scan(draft.input, null), declared)?.let { return it }
        // Past the cap the cover is not sent to moderation (review F34); the admin still sees it.
        if (cover != null && RaceCheckReason.DAILY_CAP !in reasons) reasons += coverReasons(photographerId, cover.bytes)

        val eventId = UUID.randomUUID()
        val coverKey = cover?.let { eventCoverService.upload(eventId, it.bytes, it.contentType) }
        var stored = false
        try {
            val result = transactionTemplate.execute {
                eventRepository.lockRaceCreation()
                val found = scan(draft.input, null)
                stopOutcome(found, declared)?.let { return@execute it }
                // Re-counted under the lock: a race that tips over the cap keeps its cover result too.
                if (overDailyCap(photographerId)) reasons += RaceCheckReason.DAILY_CAP
                insert(photographerId, eventId, draft, coverKey, settings.metadataPublicByDefault,
                    found.result.strong.plus(found.result.medium).firstOrNull { it.candidate.id == declared }, reasons.sorted())
            }!!
            stored = result.event != null
            return result
        } catch (ex: DataIntegrityViolationException) {
            rethrowRaceDuplicate(ex, "This race already exists. Join it instead.")
        } finally {
            // Nothing stored (a race found inside the lock, a violation, any failure): drop the cover.
            if (!stored) coverKey?.let(eventCoverService::delete)
        }
    }

    /**
     * AA-R5: the submitter edits their race until another photographer uploads. A live race checks
     * only the changed fields; a failing edit is held in `pending_change` for an admin while the race
     * keeps its live details. A PENDING race (not on sale yet) takes the edit at once and refreshes
     * its reasons. No @Transactional, as [create]: moderation and the upload run outside one.
     */
    fun update(photographerId: UUID, eventId: UUID, req: UpdateRaceRequest, cover: AdminEventService.CoverUpload?): PhotographerEventDetailDto {
        requireSwitchOn()
        accessGuard.requireApproved(photographerId, "editing a race")
        // Authority and field checks first, so a refused edit moderates and uploads nothing.
        transactionTemplate.execute { editOf(lockForEdit(eventId, photographerId), req) }
        val coverMiss = cover?.let { coverReasons(photographerId, it.bytes) }.orEmpty()
        val coverKey = cover?.let { eventCoverService.upload(eventId, it.bytes, it.contentType) }
        val obsolete = mutableListOf<String>()
        var stored = false
        try {
            val dto = transactionTemplate.execute {
                val event = lockForEdit(eventId, photographerId)
                val edit = editOf(event, req)
                val now = OffsetDateTime.now(clock)
                val live = event.reviewStatus != EventReviewStatus.PENDING
                val held = event.pendingChange.takeIf(RaceEdit::isHeld)
                val changed = edit != RaceEdit() || coverKey != null || held != null
                val duplicate = if (edit.matchChanged) duplicateNote(event, edit) else null
                val found = (editReasons(event, edit, live) + coverMiss +
                    listOfNotNull(duplicate?.let { RaceCheckReason.POSSIBLE_DUPLICATE })).distinct().sorted()
                val oldCover = event.coverS3Key
                if (!live || found.isEmpty()) {
                    if (!live) {
                        val rechecked = recheckedBy(event, edit, coverKey != null)
                        event.checkReasons = (event.checkReasons.orEmpty().filterNot { it in rechecked } + found).distinct().sorted()
                    } else {
                        event.pendingChange = null
                        event.reviewStatus = EventReviewStatus.APPROVED
                        event.checkReasons = null
                    }
                    edit.applyTo(event, matcher)
                    if (coverKey != null) {
                        event.coverS3Key = coverKey
                        oldCover?.takeIf { it.isNotBlank() }?.let(obsolete::add)
                    }
                } else {
                    // The race keeps selling on its live details; a later failing edit replaces this one.
                    event.pendingChange = edit.toJson(coverKey, photographerId, now)
                    event.reviewStatus = EventReviewStatus.CHANGE_PENDING
                    event.checkReasons = found
                    val note = (listOf("Held race edit: " + found.joinToString(", ") { it.name }) + listOfNotNull(duplicate))
                        .joinToString("; ")
                    if (!systemFlagService.raise(FlagTargetKind.EVENT, event.id, SystemFlagReason.RACE_EDIT_HELD, note = note)) {
                        systemFlagService.appendNote(FlagTargetKind.EVENT, event.id, SystemFlagReason.RACE_EDIT_HELD, "$now $note")
                    }
                    pingAdmins("event_change_requested", event.id, photographerId, now)
                }
                RaceEdit.heldCover(held)?.takeIf { it != coverKey }?.let(obsolete::add)
                if (changed) {
                    event.reviewRevision++
                    eventRepository.saveAndFlush(event)
                }
                val ep = eventPhotographerRepository.findById(EventPhotographerId(eventId, photographerId))
                    .orElseGet { EventPhotographer(id = EventPhotographerId(eventId, photographerId)) }
                detailDto(event, ep, eventDtoMapper.resolveBannerUrl(event), viewerId = photographerId).copy(canEditRace = true)
            }!!
            stored = true
            obsolete.forEach(eventCoverService::delete)
            return dto
        } catch (ex: DataIntegrityViolationException) {
            rethrowRaceDuplicate(ex, "Another race already has this name on this date.")
        } finally {
            if (!stored) coverKey?.let(eventCoverService::delete)
        }
    }

    private fun lockForEdit(eventId: UUID, photographerId: UUID): Event {
        val event = eventRepository.findRaceForSubmitterForUpdate(eventId, photographerId)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Race not found")
        if (eventPhotographerRepository.existsOtherUploader(eventId, photographerId)) {
            throw ApiException(HttpStatus.FORBIDDEN, ErrorCodes.RACE_EDIT_LOCKED,
                "Another photographer has uploaded to this race, so you can no longer change its details. Ask an admin instead.")
        }
        return event
    }

    // Only fields that differ from the live race count; blank means unchanged.
    private fun editOf(event: Event, req: UpdateRaceRequest): RaceEdit {
        fun changed(raw: String?, live: String?) = raw?.trim()?.takeIf { it.isNotEmpty() && it != live }
        val date = changed(req.date, null)?.let(EventInputs::parseDate)?.takeIf { it != event.date }
            ?.also { eventFaceExpiry.checkDateChange(event, it) }
        val region = changed(req.regionCode, event.regionCode)
        val province = changed(req.provinceCode, event.provinceCode)
        if (region != null || province != null) {
            EventInputs.validateLocation(regionsService, region ?: event.regionCode, province ?: event.provinceCode, required = true)
        }
        return RaceEdit(
            title = changed(req.title, event.name),
            date = date,
            location = changed(req.location, event.location),
            regionCode = region,
            provinceCode = province,
            eventType = changed(req.eventType, event.eventType?.name)?.let(::raceType),
            organizerName = changed(req.organizerName, event.organizerName),
            description = changed(req.description, event.description),
        )
    }

    // A live race passed its checks once, so only the changed text is checked; a pending one re-checks all of it.
    private fun editReasons(event: Event, edit: RaceEdit, live: Boolean): List<RaceCheckReason> = buildList {
        if (edit.textChanged) {
            addAll(if (live) textChecks(edit.title, edit.organizerName, edit.description)
                else textChecks(edit.title ?: event.name, edit.organizerName ?: event.organizerName, edit.description ?: event.description))
        }
        if (edit.date != null && outOfWindow(edit.date)) add(RaceCheckReason.DATE_OUT_OF_WINDOW)
    }

    // The reasons this edit re-decides on a pending race; the others (daily cap, a declared duplicate) stay.
    private fun recheckedBy(event: Event, edit: RaceEdit, newCover: Boolean): Set<RaceCheckReason> = buildSet {
        if (edit.textChanged) addAll(listOf(RaceCheckReason.BAD_WORDS, RaceCheckReason.NO_RACE_WORD, RaceCheckReason.NON_RACE_WORD))
        if (edit.date != null) add(RaceCheckReason.DATE_OUT_OF_WINDOW)
        if (newCover) addAll(listOf(RaceCheckReason.COVER_FLAGGED, RaceCheckReason.COVER_UNCHECKED))
        if (edit.matchChanged && event.possibleDuplicateOf == null) add(RaceCheckReason.POSSIBLE_DUPLICATE)
    }

    // Under the race lock, like create; a strong or medium match names the top candidate for the admin.
    private fun duplicateNote(event: Event, edit: RaceEdit): String? {
        val input = RaceInput(edit.title ?: event.name, edit.date ?: event.date, edit.provinceCode ?: event.provinceCode,
            edit.organizerName ?: event.organizerName)
        val found = lockAndFindDuplicates(input, excludeId = event.id)
        val (top, strength) = found.strong.firstOrNull()?.let { it to "strong" }
            ?: found.medium.firstOrNull()?.let { it to "medium" } ?: return null
        return "Possible duplicate of ${top.candidate.name} (${top.candidate.id}), $strength"
    }

    @Transactional
    fun join(photographerId: UUID, eventId: UUID, typedName: String?): RaceCreateResultDto {
        requireSwitchOn()
        accessGuard.requireApproved(photographerId, "joining a race")
        val allowed = SellerDetailsStatus.uploadGateSet(identityProperties.gateEnabled)
        if (!sellerDetailsRepository.existsByUserIdAndStatusIn(photographerId, allowed)) throw sellerDetailsRequired()
        val event = joinInternal(eventId, photographerId, typedName)
        val ep = eventPhotographerRepository.findById(EventPhotographerId(eventId, photographerId))
            .orElseGet { EventPhotographer(id = EventPhotographerId(eventId, photographerId)) }
        return RaceCreateResultDto(
            outcome = "joined",
            event = detailDto(event, ep, eventDtoMapper.resolveBannerUrl(event), viewerId = photographerId),
        )
    }

    /** Coverage row plus an optional alias, no switch or seller checks (AA-R6's admin "same race" reuses it). */
    fun joinInternal(eventId: UUID, photographerId: UUID, typedName: String?): Event {
        val event = eventRepository.findById(eventId).orElse(null)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        if (event.createdBy != null || event.deletedAt != null || event.reviewStatus == EventReviewStatus.REJECTED) {
            throw ApiException(HttpStatus.UNPROCESSABLE_ENTITY, ErrorCodes.EVENT_NOT_JOINABLE,
                "Only a live shared race can be joined.")
        }
        val metadataPublic = photographerSettingsRepository.findById(photographerId).orElse(null)?.metadataPublicByDefault ?: false
        eventPhotographerRepository.insertCoverageIfAbsent(eventId, photographerId, metadataPublic)
        val alias = typedName?.let(matcher::nameKey).orEmpty()
        if (alias.isNotEmpty() && alias != (event.nameKey ?: matcher.nameKey(event.name)) &&
            wordListChecker.hits(typedName!!, setOf(WordList.BAD_WORDS)).isEmpty()) {
            eventRepository.insertAlias(eventId, alias, photographerId)
        }
        return event
    }

    /** AA-R7: the admin create warning. Read-only, no lock; strong matches first, then medium ones. */
    fun findDuplicates(input: RaceInput, excludeId: UUID? = null): List<RaceCandidateDto> {
        val found = scan(input, excludeId)
        return found.result.strong.map { candidateDto(found.events.getValue(it.candidate.id), "strong") } +
            found.result.medium.map { candidateDto(found.events.getValue(it.candidate.id), "medium") }
    }

    /**
     * AA-R7 (Q11): for each PENDING Other photographer event, the top strong or medium shared race
     * within the date tolerance. One candidate read per distinct date; other rows are skipped.
     */
    fun lookalikes(rows: List<Event>): Map<UUID, RaceCandidateDto> = rows
        .filter { it.eventType == null && it.createdBy != null && it.reviewStatus == EventReviewStatus.PENDING }
        .groupBy { it.date }
        .flatMap { (date, sameDay) ->
            val near = candidatesNear(date)
            val aliases = aliasesOf(near)
            sameDay.mapNotNull { row -> top(scanAmong(inputOf(row), near, aliases))?.let { row.id to it } }
        }.toMap()

    /** AA-R7: each row's stored possible duplicate, scored against it now (aliases included). */
    fun duplicatesOf(rows: List<Event>): Map<UUID, RaceCandidateDto> {
        val pairs = rows.filter { it.possibleDuplicateOf != null }
        if (pairs.isEmpty()) return emptyMap()
        val targets = eventRepository.findAllById(pairs.mapNotNull { it.possibleDuplicateOf }.toSet()).associateBy { it.id }
        val aliases = aliasesOf(targets.values.toList())
        return pairs.mapNotNull { row ->
            val target = targets[row.possibleDuplicateOf] ?: return@mapNotNull null
            // The admin decides either way; a pair that no longer scores strong reads as medium.
            row.id to (top(scanAmong(inputOf(row), listOf(target), aliases)) ?: candidateDto(target, "medium"))
        }.toMap()
    }

    private fun top(scan: Scan): RaceCandidateDto? =
        scan.result.strong.firstOrNull()?.let { candidateDto(scan.events.getValue(it.candidate.id), "strong") }
            ?: scan.result.medium.firstOrNull()?.let { candidateDto(scan.events.getValue(it.candidate.id), "medium") }

    private fun inputOf(e: Event) = RaceInput(e.name, e.date, e.provinceCode, e.organizerName)

    /** Takes the race-creation lock, then re-checks; must run in the caller's transaction (AA-R5's edit). */
    @Transactional(propagation = Propagation.MANDATORY)
    fun lockAndFindDuplicates(input: RaceInput, excludeId: UUID? = null): DuplicateResult {
        eventRepository.lockRaceCreation()
        return scan(input, excludeId).result
    }

    private class Scan(val result: DuplicateResult, val events: Map<UUID, Event>)

    private fun scan(input: RaceInput, excludeId: UUID?): Scan {
        val events = candidatesNear(input.date).filter { it.id != excludeId }
        return scanAmong(input, events, aliasesOf(events))
    }

    private fun candidatesNear(date: LocalDate): List<Event> {
        val tolerance = raceProperties.duplicate.dateToleranceDays
        return eventRepository.findRaceCandidates(date.minusDays(tolerance), date.plusDays(tolerance))
    }

    private fun aliasesOf(events: List<Event>): Map<UUID, List<String>> = if (events.isEmpty()) emptyMap()
        else eventRepository.findAliasKeys(events.map { it.id }).groupBy({ it.getEventId() }, { it.getNameKey() })

    private fun scanAmong(input: RaceInput, events: List<Event>, aliases: Map<UUID, List<String>>): Scan {
        if (events.isEmpty()) return Scan(DuplicateResult(emptyList(), emptyList()), emptyMap())
        val result = matcher.classify(input, events.map {
            RaceCandidate(it.id, it.name, it.date, it.provinceCode, it.organizerName, aliases[it.id].orEmpty().toSet())
        })
        return Scan(result, events.associateBy { it.id })
    }

    // A match or candidates the photographer has not declared different: answer and store nothing.
    private fun stopOutcome(scan: Scan, declared: UUID?): RaceCreateResultDto? {
        val (outcome, found) = when {
            scan.result.strong.isNotEmpty() -> "match" to scan.result.strong
            scan.result.medium.isNotEmpty() -> "candidates" to scan.result.medium
            else -> return null
        }
        if (declared != null && found.any { it.candidate.id == declared }) return null
        val strength = if (outcome == "match") "strong" else "medium"
        return RaceCreateResultDto(outcome = outcome, candidates = found.map {
            val e = scan.events.getValue(it.candidate.id)
            // A race still waiting for an admin has not been looked at: its cover is not shown to the next photographer.
            candidateDto(e, strength, showCover = e.reviewStatus != EventReviewStatus.PENDING)
        })
    }

    /**
     * AA-R4 (R5 reuses it): each check passes quietly or names why an admin must look; nothing
     * here refuses. [cover] null skips moderation. The daily cap counts this photographer's races
     * since Manila midnight.
     */
    fun checksFor(input: RaceInput, cover: ByteArray?, photographerId: UUID, description: String? = null): List<RaceCheckReason> {
        val reasons = textChecks(input.name, input.organizerName, description).toMutableList()
        if (outOfWindow(input.date)) reasons += RaceCheckReason.DATE_OUT_OF_WINDOW
        if (overDailyCap(photographerId)) reasons += RaceCheckReason.DAILY_CAP
        if (cover != null && RaceCheckReason.DAILY_CAP !in reasons) reasons += coverReasons(photographerId, cover)
        return reasons
    }

    // A null text is not checked (an edit that leaves it unchanged); the race-word checks read the name only.
    private fun textChecks(name: String?, organizer: String?, description: String?): List<RaceCheckReason> = buildList {
        if (listOfNotNull(name, organizer, description).any { wordListChecker.hits(it, setOf(WordList.BAD_WORDS)).isNotEmpty() }) {
            add(RaceCheckReason.BAD_WORDS)
        }
        if (name != null) {
            if (wordListChecker.hits(name, setOf(WordList.RACE_WORDS)).isEmpty()) add(RaceCheckReason.NO_RACE_WORD)
            if (wordListChecker.hits(name, setOf(WordList.NON_RACE_WORDS)).isNotEmpty()) add(RaceCheckReason.NON_RACE_WORD)
        }
    }

    private fun outOfWindow(date: LocalDate): Boolean {
        val today = LocalDate.now(clock.withZone(EventDtoMapper.PH_ZONE))
        return date < today.minusDays(raceProperties.dateWindowPastDays) || date > today.plusDays(raceProperties.dateWindowAheadDays)
    }

    // The same per-user daily cap as profile images: past it the cover is UNCHECKED and an admin looks, no AWS call.
    private fun coverReasons(photographerId: UUID, cover: ByteArray): List<RaceCheckReason> =
        when (imageModerationService.checkFor(photographerId, cover).verdict) {
            ImageModerationVerdict.CLEAN -> emptyList()
            ImageModerationVerdict.FLAGGED -> listOf(RaceCheckReason.COVER_FLAGGED)
            ImageModerationVerdict.UNCHECKED -> listOf(RaceCheckReason.COVER_UNCHECKED)
        }

    private fun overDailyCap(photographerId: UUID): Boolean {
        val midnight = LocalDate.now(clock.withZone(EventDtoMapper.PH_ZONE)).atStartOfDay(EventDtoMapper.PH_ZONE).toOffsetDateTime()
        return eventRepository.countRacesSubmittedSince(photographerId, midnight) >= raceProperties.dailyCreateCap
    }

    private fun insert(
        photographerId: UUID, eventId: UUID, draft: Draft, coverKey: String?, metadataPublic: Boolean, declared: Scored?,
        checkMiss: List<RaceCheckReason>,
    ): RaceCreateResultDto {
        val req = draft.req
        val now = OffsetDateTime.now(clock)
        val reasons = (checkMiss + listOfNotNull(declared?.let { RaceCheckReason.POSSIBLE_DUPLICATE })).sorted()
        val live = reasons.isEmpty()
        val event = eventRepository.saveAndFlush(
            Event(
                id = eventId,
                slug = EventInputs.slugify(req.title),
                name = req.title.trim(),
                date = draft.input.date,
                location = req.location.trim(),
                status = if (live) EventStatus.ACTIVE else EventStatus.DRAFT,
                description = req.description?.trim().orEmpty(),
                organizerName = req.organizerName?.trim().orEmpty(),
                pricePerPhoto = platformProperties.photoPricePhp,
                coverS3Key = coverKey,
                regionCode = req.regionCode.trim(),
                provinceCode = req.provinceCode.trim(),
                createdAt = now,
                updatedAt = now,
                visibility = EventVisibility.PUBLIC,
                pricingMode = EventPricingMode.PAID,
                watermarkPolicy = WatermarkPolicy.PLATFORM,
                reviewStatus = if (live) EventReviewStatus.APPROVED else EventReviewStatus.PENDING,
            ).apply {
                eventType = draft.type
                submittedBy = photographerId
                nameKey = matcher.nameKey(name)
                possibleDuplicateOf = declared?.candidate?.id
                if (!live) checkReasons = reasons
            },
        )
        eventPhotographerRepository.insertCoverageIfAbsent(event.id, photographerId, metadataPublic)
        var pingAdmins = false
        if (declared != null) {
            val strength = if (declared.score >= raceProperties.duplicate.strongScore) "strong" else "medium"
            systemFlagService.raise(FlagTargetKind.EVENT, event.id, SystemFlagReason.POSSIBLE_DUPLICATE,
                note = "Possible duplicate of ${declared.candidate.name} (${declared.candidate.id}), $strength")
            pingAdmins = true
        }
        if (RaceCheckReason.DAILY_CAP in checkMiss) {
            // Past the cap: one open USER flag per photographer, not one EVENT flag per race (review F34).
            val race = "${event.name} ${event.date}"
            if (systemFlagService.raise(FlagTargetKind.USER, photographerId, SystemFlagReason.RACE_CHECK_MISS, note = "daily race cap: $race")) {
                pingAdmins = true
            } else {
                systemFlagService.appendNote(FlagTargetKind.USER, photographerId, SystemFlagReason.RACE_CHECK_MISS, "$now $race")
            }
        } else if (checkMiss.isNotEmpty()) {
            systemFlagService.raise(FlagTargetKind.EVENT, event.id, SystemFlagReason.RACE_CHECK_MISS,
                note = checkMiss.joinToString(", ") { it.name })
            pingAdmins = true
        }
        if (pingAdmins) pingAdmins("event_submitted", event.id, photographerId, now)
        val ep = EventPhotographer(id = EventPhotographerId(event.id, photographerId), metadataPublic = metadataPublic)
        return RaceCreateResultDto(
            outcome = if (live) "created" else "pending_review",
            event = detailDto(event, ep, eventDtoMapper.resolveBannerUrl(event), viewerId = photographerId),
            checkReasons = event.checkReasons.orEmpty().map { it.name },
        )
    }

    private fun pingAdmins(type: String, eventId: UUID, photographerId: UUID, now: OffsetDateTime) =
        eventPublisher.publishEvent(AdminInboxEvent(payload = mapOf(
            "type" to type,
            "entityId" to eventId.toString(),
            "actorId" to photographerId.toString(),
            "occurredAt" to now.toString(),
        )))

    private fun candidateDto(e: Event, strength: String, showCover: Boolean = true) = RaceCandidateDto(e.id, e.slug, e.name, e.date, e.location,
        e.regionCode, e.provinceCode, e.organizerName, e.eventType?.name, if (showCover) eventDtoMapper.resolveBannerUrl(e) else null, e.reviewStatus.wire,
        strength)

    private fun validate(req: CreateRaceRequest): Draft {
        val type = raceType(req.eventType)
        EventInputs.validateLocation(regionsService, req.regionCode.trim(), req.provinceCode.trim(), required = true)
        val date: LocalDate = EventInputs.parseDate(req.date)
        return Draft(RaceInput(req.title.trim(), date, req.provinceCode.trim(), req.organizerName?.trim()), type, req)
    }

    private fun requireSwitchOn() {
        if (!raceProperties.selfServeEnabled) {
            throw ApiException(HttpStatus.FORBIDDEN, ErrorCodes.SELF_SERVE_RACES_DISABLED,
                "Creating and joining race events is not open yet.")
        }
    }

    private fun sellerDetailsRequired() = ApiException(HttpStatus.FORBIDDEN, ErrorCodes.SELLER_DETAILS_REQUIRED,
        "Before you can create or join a race, finish your seller check in your photographer settings on quickpitik.com.")

    companion object {
        const val RACE_UNIQUE_INDEX = "uq_events_race_date_name"

        /** One of the six shared types; the admin create (AA-R7) parses its field the same way. */
        fun raceType(code: String): RaceType = RaceType.entries.firstOrNull { it.name == code.trim() }
            ?: throw ValidationException(code = ErrorCodes.VALIDATION_ERROR, message = "Choose one of the listed race types.",
                field = "eventType")

        /** AA-R10's partial unique index on (date, name_key), the lock's last line of defence: 409, never a 500. */
        fun rethrowRaceDuplicate(ex: DataIntegrityViolationException, message: String): Nothing {
            val violated = generateSequence<Throwable>(ex) { it.cause }
                .filterIsInstance<ConstraintViolationException>().firstOrNull()?.constraintName
            if (violated == RACE_UNIQUE_INDEX) throw ConflictException(code = ErrorCodes.RACE_DUPLICATE, message = message)
            throw ex
        }
    }
}

/** AA-R5: the rule [EventRepository.findRaceForSubmitterForUpdate] locks on, for reads that cannot lock. */
fun Event.isEditableRaceOf(photographerId: UUID): Boolean =
    eventType != null && submittedBy == photographerId && deletedAt == null && reviewStatus != EventReviewStatus.REJECTED

/**
 * AA-R5: the fields a race edit changes (null = unchanged). A held edit lives in `pending_change` as
 * `{ kind: "race_edit", fields, coverS3Key?, requestedBy, requestedAt }`, with only the changed
 * fields and never a price.
 */
data class RaceEdit(
    val title: String? = null,
    val date: LocalDate? = null,
    val location: String? = null,
    val regionCode: String? = null,
    val provinceCode: String? = null,
    val eventType: RaceType? = null,
    val organizerName: String? = null,
    val description: String? = null,
) {
    val textChanged: Boolean get() = title != null || organizerName != null || description != null

    // The fields the duplicate check reads.
    val matchChanged: Boolean get() = title != null || date != null || provinceCode != null

    /** Writes the fields and keeps `name_key` in step with the name. */
    fun applyTo(event: Event, matcher: DuplicateMatcher) {
        title?.let { event.name = it; event.nameKey = matcher.nameKey(it) }
        date?.let { event.date = it }
        location?.let { event.location = it }
        regionCode?.let { event.regionCode = it }
        provinceCode?.let { event.provinceCode = it }
        eventType?.let { event.eventType = it }
        organizerName?.let { event.organizerName = it }
        description?.let { event.description = it }
    }

    fun toJson(coverS3Key: String?, requestedBy: UUID, requestedAt: OffsetDateTime): Map<String, Any?> = mapOf(
        "kind" to KIND,
        "fields" to listOf(
            "title" to title, "date" to date?.toString(), "location" to location, "regionCode" to regionCode,
            "provinceCode" to provinceCode, "eventType" to eventType?.name, "organizerName" to organizerName,
            "description" to description,
        ).filter { it.second != null }.toMap(),
        "coverS3Key" to coverS3Key,
        "requestedBy" to requestedBy.toString(),
        "requestedAt" to requestedAt.toString(),
    )

    companion object {
        const val KIND = "race_edit"

        fun isHeld(pendingChange: Map<String, Any?>?): Boolean = pendingChange?.get("kind") == KIND

        fun heldCover(pendingChange: Map<String, Any?>?): String? =
            pendingChange?.takeIf(::isHeld)?.get("coverS3Key") as? String

        fun fromJson(json: Map<String, Any?>): RaceEdit {
            val fields = json["fields"] as? Map<*, *> ?: emptyMap<String, Any?>()
            fun text(key: String) = fields[key] as? String
            return RaceEdit(
                title = text("title"),
                date = text("date")?.let(LocalDate::parse),
                location = text("location"),
                regionCode = text("regionCode"),
                provinceCode = text("provinceCode"),
                eventType = text("eventType")?.let(RaceType::valueOf),
                organizerName = text("organizerName"),
                description = text("description"),
            )
        }
    }
}
