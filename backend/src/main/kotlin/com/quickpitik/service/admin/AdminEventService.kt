package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminEventDeleteResponseDto
import com.quickpitik.dto.admin.AdminListEventDto
import com.quickpitik.dto.admin.CreateAdminEventRequest
import com.quickpitik.dto.admin.UpdateAdminEventRequest
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.config.AiApiProperties
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.events.EventCoverService
import com.quickpitik.service.events.EventDtoMapper
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
    private val aiApiClient: AiApiClient,
    private val aiApiProperties: AiApiProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional(readOnly = true)
    fun list(
        stateFilter: String?,
        params: PaginationParams,
    ): PaginatedResponse<AdminListEventDto> {
        val pageable = OffsetLimitPageable(params)
        val page = eventRepository.pageForAdmin(
            search = "",
            dateFrom = LocalDate.of(1900, 1, 1),
            dateTo = LocalDate.of(9999, 12, 31),
            pageable = pageable,
        )
        val items = page.content
            .map { eventDtoMapper.toAdminListDto(it) }
            .let { rows ->
                if (stateFilter.isNullOrBlank()) rows
                else rows.filter { it.state == stateFilter.trim().lowercase() }
            }
        return PaginatedResponse.of(items, page.totalElements, params)
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
                bannerUrl = null,
                photoCount = 0,
                participantCount = 0,
                status = EventStatus.ACTIVE,
                description = "",
                organizerName = "",
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
        // Price-per-photo override propagates to every photo already uploaded
        // under this event so runner-facing galleries pick up the new price
        // on next paint. `compareTo` (not `!=`) because BigDecimal equality
        // also weighs scale — 125 vs 125.00 would otherwise look like a
        // change. Cart drift is surfaced to runners at checkout via the
        // existing CART_ITEM_PRICE_CHANGED flow; we deliberately do not
        // mutate cart_items.price_php_at_add.
        req.pricePerPhoto?.let { rawPrice ->
            val newPrice = validatedPrice(rawPrice)
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

    private fun parseDate(raw: String): LocalDate =
        runCatching { LocalDate.parse(raw.trim()) }.getOrElse {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "date must be ISO yyyy-MM-dd",
                field = "date",
            )
        }

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

    private fun slugify(title: String): String {
        val base = title.trim().lowercase()
            .replace(Regex("[^a-z0-9\\s-]"), "")
            .replace(Regex("\\s+"), "-")
            .replace(Regex("-+"), "-")
            .trim('-')
            .ifBlank { "event" }
        // Append a short uniqueness suffix so two events with the same
        // title don't collide on the unique index. Keeps slugs human-
        // readable while sidestepping the `slug` UNIQUE constraint.
        val suffix = UUID.randomUUID().toString().take(6)
        return "$base-$suffix"
    }
}
