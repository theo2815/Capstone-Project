package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.events.EventDetailDto
import com.quickpitik.dto.events.EventDto
import com.quickpitik.entity.EventStatus
import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.service.events.EventService
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.time.LocalDate
import java.time.format.DateTimeParseException

@RestController
@RequestMapping("/api/v1/events")
class EventController(
    private val eventService: EventService,
) {
    @GetMapping
    fun list(
        @RequestParam(required = false) status: String?,
        @RequestParam(required = false) search: String?,
        @RequestParam(required = false) city: String?,
        @RequestParam(required = false) dateFrom: String?,
        @RequestParam(required = false) dateTo: String?,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): PaginatedResponse<EventDto> {
        val pagination = PaginationParams.of(offset, limit)
        val statuses = parseStatusList(status)
        val from = parseDate(dateFrom, "dateFrom")
        val to = parseDate(dateTo, "dateTo")
        return eventService.list(statuses, search, city, from, to, pagination)
    }

    // Race-day-only viewing is enforced FE-side: /events/[slug] swaps the
    // cockpit for an "Opens on [date]" placeholder when the date is in the
    // future. The endpoint stays a plain getter so admin's "View ↗" works
    // without the SSR request needing a JWT to opt into a preview mode.
    @GetMapping("/{slug}")
    fun bySlug(@PathVariable slug: String): EventDetailDto =
        eventService.findBySlug(slug)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")

    private fun parseStatusList(raw: String?): Collection<EventStatus> {
        if (raw.isNullOrBlank()) return emptyList()
        return raw.split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { token ->
            try {
                EventStatus.valueOf(token.uppercase())
            } catch (_: IllegalArgumentException) {
                throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "Unknown event status: $token",
                    field = "status",
                )
            }
        }.filter { it != EventStatus.DRAFT }
    }

    private fun parseDate(raw: String?, field: String): LocalDate? {
        if (raw.isNullOrBlank()) return null
        return try {
            LocalDate.parse(raw)
        } catch (_: DateTimeParseException) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "Invalid ISO date for $field: $raw",
                field = field,
            )
        }
    }
}
