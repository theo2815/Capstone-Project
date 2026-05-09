package com.quickpitik.service.events

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.events.EventDetailDto
import com.quickpitik.dto.events.EventDto
import com.quickpitik.dto.events.toDetailDto
import com.quickpitik.dto.events.toListDto
import com.quickpitik.entity.EventStatus
import com.quickpitik.repository.EventRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.LocalDate

@Service
@Transactional(readOnly = true)
class EventService(
    private val eventRepository: EventRepository,
) {
    fun list(
        statuses: Collection<EventStatus>,
        search: String?,
        city: String?,
        dateFrom: LocalDate?,
        dateTo: LocalDate?,
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
            pageable = pageable,
        )
        return PaginatedResponse(
            items = page.content.map { it.toListDto() },
            total = page.totalElements,
            offset = pagination.offset,
            limit = pagination.limit,
        )
    }

    fun findBySlug(slug: String): EventDetailDto? =
        eventRepository.findBySlugAndDeletedAtIsNull(slug)?.toDetailDto()

    private companion object {
        val LIST_DEFAULT_STATUSES = listOf(EventStatus.ACTIVE, EventStatus.COMPLETED, EventStatus.ARCHIVED)
        val DATE_MIN_SENTINEL: LocalDate = LocalDate.of(1900, 1, 1)
        val DATE_MAX_SENTINEL: LocalDate = LocalDate.of(9999, 12, 31)
    }
}
