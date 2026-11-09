package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.IdentityProperties
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.RaceProperties
import com.quickpitik.dto.photographer.CreateRaceRequest
import com.quickpitik.dto.photographer.RaceCandidateDto
import com.quickpitik.dto.photographer.RaceCreateResultDto
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
        if (cover != null && RaceCheckReason.DAILY_CAP !in reasons) reasons += coverReasons(cover.bytes)

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
            val violated = generateSequence<Throwable>(ex) { it.cause }
                .filterIsInstance<ConstraintViolationException>().firstOrNull()?.constraintName
            // AA-R10's partial unique index on (date, name_key): the lock's last line of defence.
            if (violated == RACE_UNIQUE_INDEX) {
                throw ConflictException(code = ErrorCodes.RACE_DUPLICATE, message = "This race already exists. Join it instead.")
            }
            throw ex
        } finally {
            // Nothing stored (a race found inside the lock, a violation, any failure): drop the cover.
            if (!stored) coverKey?.let(eventCoverService::delete)
        }
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

    /** Read-only, no lock (AA-R7's admin hint). */
    fun findDuplicates(input: RaceInput, excludeId: UUID? = null): DuplicateResult = scan(input, excludeId).result

    /** Takes the race-creation lock, then re-checks; must run in the caller's transaction (AA-R5's edit). */
    @Transactional(propagation = Propagation.MANDATORY)
    fun lockAndFindDuplicates(input: RaceInput, excludeId: UUID? = null): DuplicateResult {
        eventRepository.lockRaceCreation()
        return scan(input, excludeId).result
    }

    private class Scan(val result: DuplicateResult, val events: Map<UUID, Event>)

    private fun scan(input: RaceInput, excludeId: UUID?): Scan {
        val tolerance = raceProperties.duplicate.dateToleranceDays
        val events = eventRepository.findRaceCandidates(input.date.minusDays(tolerance), input.date.plusDays(tolerance))
            .filter { it.id != excludeId }
        if (events.isEmpty()) return Scan(DuplicateResult(emptyList(), emptyList()), emptyMap())
        val aliases = eventRepository.findAliasKeys(events.map { it.id }).groupBy({ it.getEventId() }, { it.getNameKey() })
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
        return RaceCreateResultDto(outcome = outcome, candidates = found.map { candidateDto(scan.events.getValue(it.candidate.id), strength) })
    }

    /**
     * AA-R4 (R5 reuses it): each check passes quietly or names why an admin must look; nothing
     * here refuses. [cover] null skips moderation. The daily cap counts this photographer's races
     * since Manila midnight.
     */
    fun checksFor(input: RaceInput, cover: ByteArray?, photographerId: UUID, description: String? = null): List<RaceCheckReason> {
        val reasons = mutableListOf<RaceCheckReason>()
        val texts = listOfNotNull(input.name, input.organizerName, description)
        if (texts.any { wordListChecker.hits(it, setOf(WordList.BAD_WORDS)).isNotEmpty() }) reasons += RaceCheckReason.BAD_WORDS
        if (wordListChecker.hits(input.name, setOf(WordList.RACE_WORDS)).isEmpty()) reasons += RaceCheckReason.NO_RACE_WORD
        if (wordListChecker.hits(input.name, setOf(WordList.NON_RACE_WORDS)).isNotEmpty()) reasons += RaceCheckReason.NON_RACE_WORD
        val today = LocalDate.now(clock.withZone(EventDtoMapper.PH_ZONE))
        if (input.date < today.minusDays(raceProperties.dateWindowPastDays) ||
            input.date > today.plusDays(raceProperties.dateWindowAheadDays)) reasons += RaceCheckReason.DATE_OUT_OF_WINDOW
        if (overDailyCap(photographerId)) reasons += RaceCheckReason.DAILY_CAP
        if (cover != null && RaceCheckReason.DAILY_CAP !in reasons) reasons += coverReasons(cover)
        return reasons
    }

    private fun coverReasons(cover: ByteArray): List<RaceCheckReason> = when (imageModerationService.check(cover).verdict) {
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
        if (pingAdmins) {
            eventPublisher.publishEvent(AdminInboxEvent(payload = mapOf(
                "type" to "event_submitted",
                "entityId" to event.id.toString(),
                "actorId" to photographerId.toString(),
                "occurredAt" to now.toString(),
            )))
        }
        val ep = EventPhotographer(id = EventPhotographerId(event.id, photographerId), metadataPublic = metadataPublic)
        return RaceCreateResultDto(
            outcome = if (live) "created" else "pending_review",
            event = detailDto(event, ep, eventDtoMapper.resolveBannerUrl(event), viewerId = photographerId),
            checkReasons = event.checkReasons.orEmpty().map { it.name },
        )
    }

    private fun candidateDto(e: Event, strength: String) = RaceCandidateDto(e.id, e.slug, e.name, e.date, e.location,
        e.regionCode, e.provinceCode, e.organizerName, e.eventType?.name, eventDtoMapper.resolveBannerUrl(e), e.reviewStatus.wire,
        strength)

    private fun validate(req: CreateRaceRequest): Draft {
        val type = RaceType.entries.firstOrNull { it.name == req.eventType.trim() }
            ?: throw ValidationException(code = ErrorCodes.VALIDATION_ERROR, message = "Choose one of the listed race types.",
                field = "eventType")
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
    }
}
