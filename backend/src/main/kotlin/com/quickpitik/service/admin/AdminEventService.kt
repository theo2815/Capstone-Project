package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminEventDeleteResponseDto
import com.quickpitik.dto.admin.AdminListEventDto
import com.quickpitik.dto.admin.CreateAdminEventRequest
import com.quickpitik.dto.admin.UpdateAdminEventRequest
import com.quickpitik.dto.photos.PhotographerRef
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.config.AiApiProperties
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.service.events.EventCoverService
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.events.EventInputs
import com.quickpitik.service.photographer.PricingTrio
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional
class AdminEventService(
    private val eventRepository: EventRepository,
    private val photoRepository: PhotoRepository,
    private val adminDecisionLogService: AdminDecisionLogService,
    private val eventDtoMapper: EventDtoMapper,
    private val eventCoverService: EventCoverService,
    private val aiApiClient: FaceBibProvider,
    private val aiApiProperties: AiApiProperties,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional(readOnly = true)
    fun list(
        stateFilter: String?,
        review: String?,
        params: PaginationParams,
    ): PaginatedResponse<AdminListEventDto> {
        if (review.equals("queue", ignoreCase = true)) {
            // Photographer-owned events awaiting a decision (V46): initial
            // submissions + pricing-change requests, oldest first. Small by
            // nature, so paginated in memory like the state filter below.
            val rows = eventRepository.findByReviewStatusInAndDeletedAtIsNullOrderByCreatedAtAsc(
                listOf(EventReviewStatus.PENDING, EventReviewStatus.CHANGE_PENDING),
            )
            val owners = resolveOwners(rows)
            val items = rows.drop(params.offset).take(params.limit)
                .map { eventDtoMapper.toAdminListDto(it, owners[it.createdBy]) }
            return PaginatedResponse.of(items, rows.size.toLong(), params)
        }
        val pageable = OffsetLimitPageable(params)
        val page = eventRepository.pageForAdmin(
            search = "",
            dateFrom = LocalDate.of(1900, 1, 1),
            dateTo = LocalDate.of(9999, 12, 31),
            pageable = pageable,
        )
        val owners = resolveOwners(page.content)
        val items = page.content
            .map { eventDtoMapper.toAdminListDto(it, owners[it.createdBy]) }
            .let { rows ->
                if (stateFilter.isNullOrBlank()) rows
                else rows.filter { it.state == stateFilter.trim().lowercase() }
            }
        return PaginatedResponse.of(items, page.totalElements, params)
    }

    // ── Photographer-owned event review (V46) ────────────────────────────
    // One approve + one reject serve both queue states: a submission
    // (PENDING, DRAFT) goes live; a pricing-change request (CHANGE_PENDING on
    // a LIVE event) is applied — and only here — or dropped.

    fun approve(adminId: UUID, eventId: UUID): AdminListEventDto {
        val event = loadForReview(eventId)
        when (event.reviewStatus) {
            EventReviewStatus.PENDING -> {
                event.status = EventStatus.ACTIVE
                markReviewed(event, adminId, EventReviewStatus.APPROVED)
                val decision = adminDecisionLogService.logEventDecision(
                    adminId = adminId,
                    targetEventId = event.id,
                    decision = "event_approved",
                )
                notifyOwner(
                    event,
                    PhotographerMessageKind.EVENT_APPROVED,
                    "Your event ${event.name} is approved. Uploads are open.",
                    adminId,
                    decision.id,
                )
            }
            EventReviewStatus.CHANGE_PENDING -> {
                val change = PricingTrio.fromJson(event.pendingChange.orEmpty())
                val priceChanged = change.pricePerPhoto.compareTo(event.pricePerPhoto) != 0
                val policyChanged = change.watermarkPolicy != event.watermarkPolicy
                change.applyTo(event)
                if (priceChanged) photoRepository.updatePriceByEventId(event.id, change.pricePerPhoto)
                // A different mark makes every LIVE preview stale — send them
                // back through the watermark sweep under the new policy.
                if (policyChanged) photoRepository.resetForRewatermark(event.id)
                event.pendingChange = null
                markReviewed(event, adminId, EventReviewStatus.APPROVED)
                val decision = adminDecisionLogService.logEventDecision(
                    adminId = adminId,
                    targetEventId = event.id,
                    decision = "event_change_approved",
                    meta = change.toJson(),
                )
                notifyOwner(
                    event,
                    PhotographerMessageKind.EVENT_CHANGE_APPROVED,
                    "Your change to ${event.name} is live: ${describe(change)}.",
                    adminId,
                    decision.id,
                )
            }
            else -> throw ConflictException(code = ErrorCodes.CONFLICT, message = "This event is not awaiting review.")
        }
        eventRepository.save(event)
        return eventDtoMapper.toAdminListDto(event, resolveOwners(listOf(event))[event.createdBy])
    }

    fun reject(adminId: UUID, eventId: UUID, reason: String): AdminListEventDto {
        val event = loadForReview(eventId)
        val note = reason.trim().take(500)
        when (event.reviewStatus) {
            EventReviewStatus.PENDING -> {
                markReviewed(event, adminId, EventReviewStatus.REJECTED)
                event.reviewNote = note
                val decision = adminDecisionLogService.logEventDecision(
                    adminId = adminId,
                    targetEventId = event.id,
                    decision = "event_rejected",
                    reason = note,
                )
                notifyOwner(
                    event,
                    PhotographerMessageKind.EVENT_REJECTED,
                    "Your event ${event.name} wasn't approved. Reason: $note",
                    adminId,
                    decision.id,
                )
            }
            EventReviewStatus.CHANGE_PENDING -> {
                // The live trio was never touched — just drop the request.
                event.pendingChange = null
                markReviewed(event, adminId, EventReviewStatus.APPROVED)
                event.reviewNote = note
                val decision = adminDecisionLogService.logEventDecision(
                    adminId = adminId,
                    targetEventId = event.id,
                    decision = "event_change_rejected",
                    reason = note,
                )
                notifyOwner(
                    event,
                    PhotographerMessageKind.EVENT_CHANGE_REJECTED,
                    "Your change to ${event.name} wasn't approved. Reason: $note",
                    adminId,
                    decision.id,
                )
            }
            else -> throw ConflictException(code = ErrorCodes.CONFLICT, message = "This event is not awaiting review.")
        }
        eventRepository.save(event)
        return eventDtoMapper.toAdminListDto(event, resolveOwners(listOf(event))[event.createdBy])
    }

    // Locked read: a second admin deciding the same row waits here and then
    // hits the 409 above instead of double-logging + double-notifying.
    private fun loadForReview(eventId: UUID): Event =
        eventRepository.findByIdForReview(eventId)?.takeIf { it.deletedAt == null }
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")

    private fun markReviewed(event: Event, adminId: UUID, status: EventReviewStatus) {
        event.reviewStatus = status
        event.reviewedAt = OffsetDateTime.now()
        event.reviewedBy = adminId
    }

    private fun notifyOwner(event: Event, kind: PhotographerMessageKind, body: String, adminId: UUID, decisionId: UUID) {
        val owner = event.createdBy ?: return
        adminDecisionLogService.pushMessage(
            photographerId = owner,
            kind = kind,
            body = body,
            sourceAdminId = adminId,
            sourceDecisionId = decisionId,
        )
    }

    private fun describe(change: PricingTrio): String = when (change.pricingMode) {
        com.quickpitik.entity.EventPricingMode.FREE ->
            "free, ${if (change.watermarkPolicy == com.quickpitik.entity.WatermarkPolicy.NONE) "no watermark" else "your logo"}"
        com.quickpitik.entity.EventPricingMode.PAID -> "paid at PHP ${change.pricePerPhoto.toPlainString()}"
    }

    // Owner attribution for the admin list, two IN queries per page — same
    // batch shape as PhotoService.resolvePhotographers.
    private fun resolveOwners(events: List<Event>): Map<UUID?, PhotographerRef> {
        val ids = events.mapNotNullTo(mutableSetOf()) { it.createdBy }
        if (ids.isEmpty()) return emptyMap()
        val handles = photographerSettingsRepository.findAllById(ids).associate { it.userId to it.handle }
        val names = userRepository.findAllById(ids).associate { it.id to it.name }
        return ids.associateWith { PhotographerRef(handle = handles[it], name = names[it]) }
    }

    fun create(
        adminId: UUID,
        req: CreateAdminEventRequest,
        cover: CoverUpload? = null,
    ): AdminListEventDto {
        val date = parseDate(req.date)
        val slug = slugify(req.title)
        val price = validatedPrice(req.pricePerPhoto)
        // Persist the row first so the cover key can scope to the new
        // event id (events/{id}/cover/{uuid}.jpg). Saves stay in the same
        // @Transactional method — a cover-decode failure rolls back the
        // event row.
        val event = eventRepository.save(
            Event(
                slug = slug,
                name = req.title.trim(),
                date = date,
                location = req.location.trim(),
                photoCount = 0,
                participantCount = 0,
                status = EventStatus.ACTIVE,
                description = req.description?.trim().orEmpty(),
                organizerName = req.organizerName?.trim().orEmpty(),
                pricePerPhoto = price,
                createdAt = OffsetDateTime.now(),
                updatedAt = OffsetDateTime.now(),
            ),
        )
        if (cover != null) {
            event.coverS3Key = eventCoverService.upload(event.id, cover.bytes, cover.contentType)
            eventRepository.save(event)
        }
        adminDecisionLogService.logEventDecision(
            adminId = adminId,
            targetEventId = event.id,
            decision = "event_created",
            meta = mapOf(
                "title" to req.title,
                "date" to req.date,
                "location" to req.location,
                "pricePerPhoto" to price.toPlainString(),
            ),
        )
        return eventDtoMapper.toAdminListDto(event)
    }

    /** Raw image bytes + content type for cover uploads. */
    data class CoverUpload(val bytes: ByteArray, val contentType: String?)

    fun update(
        adminId: UUID,
        eventId: UUID,
        req: UpdateAdminEventRequest,
        cover: CoverUpload? = null,
        removeCover: Boolean = false,
    ): AdminListEventDto {
        val event = eventRepository.findById(eventId).orElseThrow {
            NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        }
        val changes = mutableMapOf<String, Any?>()
        val before = mutableMapOf<String, Any?>()
        val after = mutableMapOf<String, Any?>()
        req.title?.takeIf { it.isNotBlank() }?.let {
            val newName = it.trim()
            if (event.name != newName) {
                changes["title"] = mapOf("from" to event.name, "to" to newName)
                before["title"] = event.name
                after["title"] = newName
                event.name = newName
            }
        }
        req.date?.takeIf { it.isNotBlank() }?.let {
            val parsed = parseDate(it)
            if (event.date != parsed) {
                changes["date"] = mapOf("from" to event.date.toString(), "to" to parsed.toString())
                before["date"] = event.date.toString()
                after["date"] = parsed.toString()
                event.date = parsed
            }
        }
        req.location?.takeIf { it.isNotBlank() }?.let {
            val newLocation = it.trim()
            if (event.location != newLocation) {
                changes["location"] = mapOf("from" to event.location, "to" to newLocation)
                before["location"] = event.location
                after["location"] = newLocation
                event.location = newLocation
            }
        }
        // Organizer + race-day notes feed the runner-facing "About this race"
        // strip. Same null/blank-means-no-change contract as title/location.
        req.organizerName?.takeIf { it.isNotBlank() }?.let {
            val newOrganizer = it.trim()
            if (event.organizerName != newOrganizer) {
                changes["organizerName"] = mapOf("from" to event.organizerName, "to" to newOrganizer)
                before["organizerName"] = event.organizerName
                after["organizerName"] = newOrganizer
                event.organizerName = newOrganizer
            }
        }
        req.description?.takeIf { it.isNotBlank() }?.let {
            val newDescription = it.trim()
            if (event.description != newDescription) {
                changes["description"] = mapOf("from" to event.description, "to" to newDescription)
                before["description"] = event.description
                after["description"] = newDescription
                event.description = newDescription
            }
        }
        // Price-per-photo override propagates to every photo already uploaded
        // under this event so runner-facing galleries pick up the new price
        // on next paint. `compareTo` (not `!=`) because BigDecimal equality
        // also weighs scale — 125 vs 125.00 would otherwise look like a
        // change. Existing carts need no fix-up: CartService renders the live
        // photos.price_php, which is also what OrderService.create charges.
        req.pricePerPhoto?.let { rawPrice ->
            val newPrice = validatedPrice(rawPrice)
            // A FREE photographer event (V46) has no price; the trio only
            // changes through an approved pricing request.
            if (event.isFree && newPrice.signum() != 0) {
                throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "This is a free event — approve a pricing change from the event requests queue instead.",
                    field = "pricePerPhoto",
                )
            }
            if (event.pricePerPhoto.compareTo(newPrice) != 0) {
                val oldPriceStr = event.pricePerPhoto.toPlainString()
                val newPriceStr = newPrice.toPlainString()
                changes["pricePerPhoto"] = mapOf("from" to oldPriceStr, "to" to newPriceStr)
                before["pricePerPhoto"] = oldPriceStr
                after["pricePerPhoto"] = newPriceStr
                event.pricePerPhoto = newPrice
                photoRepository.updatePriceByEventId(event.id, newPrice)
            }
        }
        // Cover handling — upload wins over remove when both are signalled.
        // Audit log records the high-level transition (set/none) rather than
        // the S3 key so the admin-overrides surface stays readable.
        if (cover != null) {
            val oldKey = event.coverS3Key
            event.coverS3Key = eventCoverService.upload(event.id, cover.bytes, cover.contentType)
            val fromState = if (oldKey.isNullOrBlank()) "none" else "set"
            changes["cover"] = mapOf("from" to fromState, "to" to "set")
            before["cover"] = fromState
            after["cover"] = "set"
            if (!oldKey.isNullOrBlank()) eventCoverService.delete(oldKey)
        } else if (removeCover && !event.coverS3Key.isNullOrBlank()) {
            val oldKey = event.coverS3Key!!
            event.coverS3Key = null
            changes["cover"] = mapOf("from" to "set", "to" to "none")
            before["cover"] = "set"
            after["cover"] = "none"
            eventCoverService.delete(oldKey)
        }
        if (changes.isEmpty()) {
            return eventDtoMapper.toAdminListDto(event)
        }
        // Append the per-row override entry per Q-A3 — replace the list reference
        // (rather than mutating in place) so Hibernate's dirty-checking notices
        // the change on the @JdbcTypeCode(SqlTypes.JSON) column.
        val entry = mapOf(
            "at" to OffsetDateTime.now().toString(),
            "adminId" to adminId.toString(),
            "before" to before.toMap(),
            "after" to after.toMap(),
        )
        event.adminOverrides = event.adminOverrides + entry
        eventRepository.save(event)
        adminDecisionLogService.logEventDecision(
            adminId = adminId,
            targetEventId = event.id,
            decision = "event_updated",
            meta = changes.toMap(),
        )
        return eventDtoMapper.toAdminListDto(event)
    }

    fun delete(adminId: UUID, eventId: UUID): AdminEventDeleteResponseDto {
        val event = eventRepository.findById(eventId).orElse(null)
            ?: return AdminEventDeleteResponseDto(removed = false)
        if (event.deletedAt != null) {
            return AdminEventDeleteResponseDto(removed = false)
        }
        event.deletedAt = OffsetDateTime.now()
        eventRepository.save(event)
        // GDPR: erase the biometric face embeddings ai-api holds for this event's
        // photos. One event-scoped bulk call — not a per-photo loop — so deleting
        // an event with thousands of photos doesn't fan out into thousands of HTTP
        // calls. Best-effort: a failure must not block the delete (the
        // orphan-person reaper is the backstop). Skipped when ai-api is disabled.
        if (aiApiProperties.enabled) {
            runCatching { aiApiClient.deleteFacesByEvent(eventId) }
                .onFailure { log.warn("ai-api face erasure failed for deleted event {}: {}", eventId, it.message) }
        }
        adminDecisionLogService.logEventDecision(
            adminId = adminId,
            targetEventId = event.id,
            decision = "event_deleted",
        )
        return AdminEventDeleteResponseDto(removed = true)
    }

    // Re-drives AI indexing for an event's photos by resetting them to PENDING
    // with a fresh attempt budget; the reconcile sweep picks them up within a
    // minute. Default scope = FAILED + PARTIAL (outage recovery — the
    // 2026-08-25 incident needed manual SQL for exactly this). all=true also
    // requeues INDEXED + SKIPPED: for provider flips (the V33 indexed_provider
    // stamp marks stale rows) and for photos uploaded while AI was disabled.
    fun reindexPhotos(adminId: UUID, eventId: UUID, all: Boolean): Int {
        val event = eventRepository.findById(eventId).orElse(null)?.takeIf { it.deletedAt == null }
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        val statuses = if (all) {
            listOf(IndexingStatus.FAILED, IndexingStatus.PARTIAL, IndexingStatus.INDEXED, IndexingStatus.SKIPPED)
        } else {
            listOf(IndexingStatus.FAILED, IndexingStatus.PARTIAL)
        }
        val requeued = photoRepository.requeueIndexing(eventId, statuses)
        log.info("Admin {} requeued {} photo(s) for indexing on event {} (all={})", adminId, requeued, event.id, all)
        return requeued
    }

    private fun parseDate(raw: String): LocalDate = EventInputs.parseDate(raw)

    // Admin-set per-photo price must be a non-negative peso amount. We
    // strip trailing zeros and clamp to scale 2 so the audit log entries
    // stay readable (avoid "125.00000" creeping in from FormData round-trips).
    private fun validatedPrice(raw: BigDecimal): BigDecimal {
        if (raw < BigDecimal.ZERO) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "pricePerPhoto must be ≥ 0",
                field = "pricePerPhoto",
            )
        }
        return raw.setScale(2, java.math.RoundingMode.HALF_UP)
    }

    // Shared with the photographer create path (V46) — see EventInputs.
    private fun slugify(title: String): String = EventInputs.slugify(title)
}
