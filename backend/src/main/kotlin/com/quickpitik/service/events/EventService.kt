package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.events.EventDetailDto
import com.quickpitik.dto.events.EventDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.LocalDate

@Service
@Transactional(readOnly = true)
class EventService(
    private val eventRepository: EventRepository,
    private val eventDtoMapper: EventDtoMapper,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
) {
    // Date filtering for "race-day-only viewing" lives on the FE: upcoming
    // events stay visible as cards on /events and the /events/[slug] page
    // renders an "Opens on [date]" placeholder instead of the full cockpit
    // until race day. The backend is just a data source — keeping the API
    // free of viewer-role plumbing also fixes admin preview (the SSR
    // request has no JWT to opt into a "show upcoming" path).
    fun list(
        statuses: Collection<EventStatus>,
        search: String?,
        city: String?,
        dateFrom: LocalDate?,
        dateTo: LocalDate?,
        includeListed: Boolean,
        pagination: PaginationParams,
    ): PaginatedResponse<EventDto> {
        val effectiveStatuses = statuses.ifEmpty { LIST_DEFAULT_STATUSES }
        val pageable = OffsetLimitPageable(pagination)
        val page = eventRepository.search(
            statuses = effectiveStatuses,
            search = search?.takeIf { it.isNotBlank() }.orEmpty(),
            city = city?.takeIf { it.isNotBlank() }.orEmpty(),
            dateFrom = dateFrom ?: DATE_MIN_SENTINEL,
            dateTo = dateTo ?: DATE_MAX_SENTINEL,
            includeListed = includeListed,
            pageable = pageable,
        )
        return PaginatedResponse(
            items = eventDtoMapper.toListDtos(page.content),
            total = page.totalElements,
            offset = pagination.offset,
            limit = pagination.limit,
        )
    }

    // Public detail (V46): drafts are not pages; unlisted events are reachable
    // by link. The owner's handle rides along so a free gallery credits them.
    fun findBySlug(slug: String): EventDetailDto? {
        val event = eventRepository.findPublicBySlug(slug) ?: return null
        val ownerHandle = event.createdBy?.let { photographerSettingsRepository.findById(it).orElse(null)?.handle }
        return eventDtoMapper.toDetailDto(event, ownerHandle)
    }

    // The public photo routes need the same "public event or 404" resolution
    // three times over and a controller may not touch a repository
    // (backend/AGENTS.md rule 1). Callers use the id only — open-in-view is
    // off, so the returned entity is detached.
    fun requirePublicBySlug(slug: String): Event = eventRepository.findPublicBySlug(slug)
        ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")

    private companion object {
        val LIST_DEFAULT_STATUSES = listOf(EventStatus.ACTIVE, EventStatus.COMPLETED, EventStatus.ARCHIVED)
        val DATE_MIN_SENTINEL: LocalDate = LocalDate.of(1900, 1, 1)
        val DATE_MAX_SENTINEL: LocalDate = LocalDate.of(9999, 12, 31)
    }
}
