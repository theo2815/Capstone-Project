package com.quickpitik.service.admin

import com.quickpitik.common.IdempotencyKey
import com.quickpitik.entity.AdminActionOperation
import com.quickpitik.entity.Role
import com.quickpitik.exception.ApiException
import org.springframework.http.HttpStatus
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
import com.quickpitik.service.reference.RegionsService
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
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
    private val transactionTemplate: TransactionTemplate,
    private val commands: AdminActionCommandService,
    private val regionsService: RegionsService,
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

    fun approve(adminId: UUID, eventId: UUID, expectedRevision: Long? = null): AdminListEventDto {
        // Only a pricing change writes photo rows, so a new submission leaves them
        // unlocked for uploads and trash. This read holds no lock: the state is
        // checked again below, under the event lock.
        val photosLocked = eventRepository.findReviewStatus(eventId) == EventReviewStatus.CHANGE_PENDING
        if (photosLocked) lockEventPhotos(eventId)
        val event = loadForReview(eventId)
        event.requireReviewRevision(expectedRevision)
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
                // Became a pricing change after the unlocked read. Locking photos now
                // would invert the photos -> events order, so the admin retries.
                if (!photosLocked) throw ConflictException(code = ErrorCodes.INVALID_STATE_TRANSITION,
                    message = "This event changed while it was being approved. Refresh and try again.")
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

    fun reject(adminId: UUID, eventId: UUID, reason: String, expectedRevision: Long? = null): AdminListEventDto {
        val event = loadForReview(eventId)
        event.requireReviewRevision(expectedRevision)
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
        event.reviewRevision++
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

    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    fun create(
        adminId: UUID,
        req: CreateAdminEventRequest,
        cover: CoverUpload? = null,
        key: IdempotencyKey? = null,
    ): AdminListEventDto {
        val normalized = req.copy(regionCode = req.regionCode?.trim(), provinceCode = req.provinceCode?.trim())
        validateLocation(normalized.regionCode, normalized.provinceCode)
        if (key != null) return createCommand(adminId, normalized, cover, key)
        val date = parseDate(req.date)
        val price = validatedPrice(req.pricePerPhoto)
        val eventId = UUID.randomUUID()
        val coverKey = cover?.let { eventCoverService.upload(eventId, it.bytes, it.contentType) }
        // ponytail: unkeyed legacy commit failure can orphan a cover; keyed callers use durable provenance.
        val result = transactionTemplate.execute { createEvent(adminId, normalized, eventId, coverKey, date, price).first }!!
        return eventDtoMapper.toAdminListDto(result)
    }

    private fun createEvent(adminId: UUID, req: CreateAdminEventRequest, eventId: UUID, coverKey: String?,
        date: LocalDate, price: BigDecimal): Pair<Event, UUID> {
        val event = eventRepository.save(
            Event(
                id = eventId,
                coverS3Key = coverKey,
                slug = slugify(req.title),
                name = req.title.trim(),
                date = date,
                location = req.location.trim(),
                regionCode = req.regionCode,
                provinceCode = req.provinceCode,
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
        val decision = adminDecisionLogService.logEventDecision(
            adminId = adminId,
            targetEventId = event.id,
            decision = "event_created",
            meta = mapOf(
                "title" to req.title,
                "date" to req.date,
                "location" to req.location,
                "regionCode" to req.regionCode,
                "provinceCode" to req.provinceCode,
                "pricePerPhoto" to price.toPlainString(),
            ),
        )
        return event to decision.id
    }

    /** Raw image bytes + content type for cover uploads. */
    data class CoverUpload(val bytes: ByteArray, val contentType: String?)

    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    fun update(
        adminId: UUID,
        eventId: UUID,
        req: UpdateAdminEventRequest,
        cover: CoverUpload? = null,
        removeCover: Boolean = false,
        key: IdempotencyKey? = null,
    ): AdminListEventDto {
        val normalized = req.copy(regionCode = req.regionCode?.trim(), provinceCode = req.provinceCode?.trim())
        validateLocation(normalized.regionCode, normalized.provinceCode)
        if (key != null) return updateCommand(adminId, eventId, normalized, cover, removeCover, key)
        // Preserve validation before the PUT, then reload in the write transaction
        // so edits made during the upload are not overwritten by a detached snapshot.
        val coverKey = cover?.let {
            val event = loadForUpdate(eventId)
            event.requireReviewRevision(req.expectedRevision)
            req.date?.takeIf { date -> date.isNotBlank() }?.let(::parseDate)
            req.pricePerPhoto?.let { price -> validatedUpdatePrice(price, event.isFree) }
            eventCoverService.upload(eventId, it.bytes, it.contentType)
        }
        val (result, oldCoverKey) = transactionTemplate.execute {
            if (req.pricePerPhoto != null) lockEventPhotos(eventId)
            val event = loadForReview(eventId)
            event.requireReviewRevision(req.expectedRevision)
            val oldKey = event.coverS3Key
            updateEvent(adminId, event, normalized, coverKey, removeCover)
            event to oldKey?.takeIf { it.isNotBlank() && it != event.coverS3Key }
        } ?: error("Event update returned null")
        // Never delete the stored cover until its replacement/removal has committed.
        oldCoverKey?.let(eventCoverService::delete)
        return eventDtoMapper.toAdminListDto(result)
    }

    private fun validateLocation(region: String?, province: String?) {
        if (region == null && province == null) return
        if (region.isNullOrBlank() || province.isNullOrBlank() || region.length > 20 || province.length > 40 ||
            !regionsService.isValidPair(region, province)) {
            throw ValidationException(code = ErrorCodes.INVALID_REGION,
                message = "Unknown regionCode/provinceCode pair", field = "regionCode")
        }
    }

    private fun authorizeCommand(adminId: UUID) {
        val actor = userRepository.findById(adminId).orElse(null)
        if (actor == null || actor.role != Role.ADMIN || actor.suspendedAt != null)
            throw ApiException(HttpStatus.FORBIDDEN,
                ErrorCodes.FORBIDDEN, "An active admin is required")
    }

    private fun createCommand(adminId: UUID, req: CreateAdminEventRequest, cover: CoverUpload?,
        key: IdempotencyKey): AdminListEventDto {
        // Cheap input checks before the cover is hashed or decoded; the claim checks them again.
        parseDate(req.date); validatedPrice(req.pricePerPhoto)
        // Omitted pairs keep the fingerprint of receipts created before location support.
        val fields = listOf(req.title.trim(), req.date.trim(), req.location.trim(),
            req.pricePerPhoto.setScale(2, java.math.RoundingMode.HALF_UP).toPlainString(),
            req.organizerName?.trim().orEmpty(), req.description?.trim().orEmpty(), cover?.let { eventCoverService.digest(it.bytes) }.orEmpty()) +
            listOfNotNull(req.regionCode, req.provinceCode)
        var processed: ByteArray? = null
        val result = commands.executeEvent(key, adminId, AdminActionOperation.EVENT_CREATE,
            null, commands.inputHash(fields), { authorizeCommand(adminId) }, {
                parseDate(req.date); validatedPrice(req.pricePerPhoto)
            }, {
                processed = cover?.let { eventCoverService.prepare(it.bytes, it.contentType) }
                processed?.let(eventCoverService::digest)
            }, { storageKey, recovery ->
                if (recovery) eventCoverService.recover(storageKey, processed!!, eventCoverService.digest(processed!!))
                else eventCoverService.uploadPrepared(storageKey, processed!!)
            }, eventCoverService::delete, { eventId, coverKey ->
                val (event, decision) = createEvent(adminId, req, eventId, coverKey, parseDate(req.date), validatedPrice(req.pricePerPhoto))
                AdminActionResult(event.id, decision, null)
            })
        return commandEventDto(result.resourceId!!)
    }

    private fun updateCommand(adminId: UUID, eventId: UUID, req: UpdateAdminEventRequest, cover: CoverUpload?,
        removeCover: Boolean, key: IdempotencyKey): AdminListEventDto {
        val fields = listOf(eventId.toString(), req.title?.trim().orEmpty(), req.date?.trim().orEmpty(),
            req.location?.trim().orEmpty(), req.pricePerPhoto?.setScale(2, java.math.RoundingMode.HALF_UP)?.toPlainString().orEmpty(),
            req.organizerName?.trim().orEmpty(), req.description?.trim().orEmpty(), req.expectedRevision?.toString().orEmpty(),
            (removeCover && cover == null).toString(), cover?.let { eventCoverService.digest(it.bytes) }.orEmpty()) +
            listOfNotNull(req.regionCode, req.provinceCode)
        var processed: ByteArray? = null
        var oldCover: String? = null
        val result = commands.executeEvent(key, adminId, AdminActionOperation.EVENT_UPDATE,
            eventId, commands.inputHash(fields), { authorizeCommand(adminId) }, {
                val event = loadForReview(eventId)
                event.requireReviewRevision(req.expectedRevision)
                req.date?.takeIf(String::isNotBlank)?.let(::parseDate)
                req.pricePerPhoto?.let { validatedUpdatePrice(it, event.isFree) }
            }, {
                processed = cover?.let { eventCoverService.prepare(it.bytes, it.contentType) }
                processed?.let(eventCoverService::digest)
            }, { storageKey, recovery ->
                if (recovery) eventCoverService.recover(storageKey, processed!!, eventCoverService.digest(processed!!))
                else eventCoverService.uploadPrepared(storageKey, processed!!)
            }, eventCoverService::delete, { _, coverKey ->
                if (req.pricePerPhoto != null) lockEventPhotos(eventId)
                val event = loadForReview(eventId)
                event.requireReviewRevision(req.expectedRevision)
                val previous = event.coverS3Key
                val decision = updateEvent(adminId, event, req, coverKey, removeCover)
                oldCover = previous?.takeIf { it.isNotBlank() && it != event.coverS3Key }
                AdminActionResult(eventId, decision, null)
            })
        oldCover?.let(eventCoverService::delete)
        return commandEventDto(result.resourceId!!)
    }

    private fun commandEventDto(eventId: UUID): AdminListEventDto {
        val event = eventRepository.findById(eventId).orElseThrow {
            NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Original event not found")
        }
        return eventDtoMapper.toAdminListDto(event, resolveOwners(listOf(event))[event.createdBy])
    }

    private fun lockEventPhotos(eventId: UUID) {
        // Match photo writers: photos before event counters, avoiding a repricing deadlock.
        photoRepository.lockIdsByEventIdForUpdate(eventId)
    }

    private fun loadForUpdate(eventId: UUID): Event =
        eventRepository.findById(eventId).orElse(null)?.takeIf { it.deletedAt == null }
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")

    private fun updateEvent(
        adminId: UUID,
        event: Event,
        req: UpdateAdminEventRequest,
        coverKey: String?,
        removeCover: Boolean,
    ): UUID? {
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
        req.regionCode?.let {
            if (event.regionCode != it) {
                changes["regionCode"] = mapOf("from" to event.regionCode, "to" to it)
                before["regionCode"] = event.regionCode
                after["regionCode"] = it
                event.regionCode = it
            }
        }
        req.provinceCode?.let {
            if (event.provinceCode != it) {
                changes["provinceCode"] = mapOf("from" to event.provinceCode, "to" to it)
                before["provinceCode"] = event.provinceCode
                after["provinceCode"] = it
                event.provinceCode = it
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
            val newPrice = validatedUpdatePrice(rawPrice, event.isFree)
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
        if (coverKey != null) {
            val oldKey = event.coverS3Key
            event.coverS3Key = coverKey
            val fromState = if (oldKey.isNullOrBlank()) "none" else "set"
            changes["cover"] = mapOf("from" to fromState, "to" to "set")
            before["cover"] = fromState
            after["cover"] = "set"
        } else if (removeCover && !event.coverS3Key.isNullOrBlank()) {
            event.coverS3Key = null
            changes["cover"] = mapOf("from" to "set", "to" to "none")
            before["cover"] = "set"
            after["cover"] = "none"
        }
        if (changes.isEmpty()) {
            return null
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
        event.reviewRevision++
        event.adminOverrides = event.adminOverrides + entry
        eventRepository.save(event)
        val decision = adminDecisionLogService.logEventDecision(
            adminId = adminId,
            targetEventId = event.id,
            decision = "event_updated",
            meta = changes.toMap(),
        )
        return decision.id
    }

    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    fun delete(adminId: UUID, eventId: UUID): AdminEventDeleteResponseDto {
        val removed = transactionTemplate.execute {
            val event = eventRepository.findByIdForReview(eventId)
                ?: return@execute false
            if (event.deletedAt != null) return@execute false
            event.deletedAt = OffsetDateTime.now()
            event.reviewRevision++
            eventRepository.save(event)
            adminDecisionLogService.logEventDecision(
                adminId = adminId,
                targetEventId = event.id,
                decision = "event_deleted",
            )
            true
        } == true
        if (!removed) return AdminEventDeleteResponseDto(removed = false)
        // GDPR: erase the biometric face embeddings ai-api holds for this event's
        // photos. One event-scoped bulk call — not a per-photo loop — so deleting
        // an event with thousands of photos doesn't fan out into thousands of HTTP
        // calls. Best-effort: a failure must not block the delete (the
        // orphan-person reaper is the backstop). Skipped when ai-api is disabled.
        if (aiApiProperties.enabled) {
            runCatching { aiApiClient.deleteFacesByEvent(eventId) }
                .onFailure { log.warn("ai-api face erasure failed for deleted event {}: {}", eventId, it.message) }
        }
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

    private fun validatedUpdatePrice(raw: BigDecimal, isFree: Boolean): BigDecimal {
        val price = validatedPrice(raw, allowZero = isFree)
        // A FREE event's pricing trio changes only through an approved request.
        if (isFree && price.signum() != 0) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "This is a free event — approve a pricing change from the event requests queue instead.",
                field = "pricePerPhoto",
            )
        }
        return price
    }

    private fun parseDate(raw: String): LocalDate = EventInputs.parseDate(raw)

    // Admin-set per-photo price must be a non-negative peso amount. We
    // strip trailing zeros and clamp to scale 2 so the audit log entries
    // stay readable (avoid "125.00000" creeping in from FormData round-trips).
    // Admin events are PAID, and a PAID event at ₱0 is unsellable (checkout
    // needs a positive price) yet not free (isFree keys off pricingMode).
    // Only a FREE photographer event may carry 0.
    private fun validatedPrice(raw: BigDecimal, allowZero: Boolean = false): BigDecimal {
        if (raw.signum() < 0 || (raw.signum() == 0 && !allowZero)) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = if (allowZero) "pricePerPhoto must be ≥ 0" else "pricePerPhoto must be greater than 0",
                field = "pricePerPhoto",
            )
        }
        return raw.setScale(2, java.math.RoundingMode.HALF_UP)
    }

    // Shared with the photographer create path (V46) — see EventInputs.
    private fun slugify(title: String): String = EventInputs.slugify(title)
}
