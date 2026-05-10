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
import com.quickpitik.repository.EventRepository
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
    private val adminDecisionLogService: AdminDecisionLogService,
) {

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
            .map { it.toAdminListDto() }
            .let { rows ->
                if (stateFilter.isNullOrBlank()) rows
                else rows.filter { it.state == stateFilter.trim().lowercase() }
            }
        return PaginatedResponse.of(items, page.totalElements, params)
    }

    fun create(adminId: UUID, req: CreateAdminEventRequest): AdminListEventDto {
        val date = parseDate(req.date)
        val slug = slugify(req.title)
        val event = eventRepository.save(
            Event(
                slug = slug,
                name = req.title.trim(),
                date = date,
                location = req.location.trim(),
                bannerUrl = req.bannerUrl,
                photoCount = 0,
                participantCount = 0,
                status = EventStatus.ACTIVE,
                description = "",
                organizerName = "",
                pricePerPhoto = BigDecimal("125"),
                createdAt = OffsetDateTime.now(),
                updatedAt = OffsetDateTime.now(),
            ),
        )
        adminDecisionLogService.logEventDecision(
            adminId = adminId,
            targetEventId = event.id,
            decision = "event_created",
            meta = mapOf("title" to req.title, "date" to req.date, "location" to req.location),
        )
        return event.toAdminListDto()
    }

    fun update(adminId: UUID, eventId: UUID, req: UpdateAdminEventRequest): AdminListEventDto {
        val event = eventRepository.findById(eventId).orElseThrow {
            NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        }
        val changes = mutableMapOf<String, Any?>()
        req.title?.takeIf { it.isNotBlank() }?.let {
            if (event.name != it) {
                changes["title"] = mapOf("from" to event.name, "to" to it)
                event.name = it.trim()
            }
        }
        req.date?.takeIf { it.isNotBlank() }?.let {
            val parsed = parseDate(it)
            if (event.date != parsed) {
                changes["date"] = mapOf("from" to event.date.toString(), "to" to parsed.toString())
                event.date = parsed
            }
        }
        req.location?.takeIf { it.isNotBlank() }?.let {
            if (event.location != it) {
                changes["location"] = mapOf("from" to event.location, "to" to it)
                event.location = it.trim()
            }
        }
        if (changes.isEmpty()) {
            return event.toAdminListDto()
        }
        eventRepository.save(event)
        adminDecisionLogService.logEventDecision(
            adminId = adminId,
            targetEventId = event.id,
            decision = "event_updated",
            meta = changes.toMap(),
        )
        return event.toAdminListDto()
    }

    fun delete(adminId: UUID, eventId: UUID): AdminEventDeleteResponseDto {
        val event = eventRepository.findById(eventId).orElse(null)
            ?: return AdminEventDeleteResponseDto(removed = false)
        if (event.deletedAt != null) {
            return AdminEventDeleteResponseDto(removed = false)
        }
        event.deletedAt = OffsetDateTime.now()
        eventRepository.save(event)
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
