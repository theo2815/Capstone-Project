package com.quickpitik.service.cart

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.cart.SavedEventSummaryDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.SavedEvent
import com.quickpitik.entity.SavedEventId
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.SavedEventRepository
import com.quickpitik.service.events.EventDtoMapper
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional
class SavedEventsService(
    private val savedEventRepository: SavedEventRepository,
    private val eventRepository: EventRepository,
    private val eventDtoMapper: EventDtoMapper,
) {
    @Transactional(readOnly = true)
    fun list(userId: UUID): List<SavedEventSummaryDto> {
        val saves = savedEventRepository.findByUserIdOrderedBySavedAtDesc(userId)
        if (saves.isEmpty()) return emptyList()
        val events = eventRepository
            .findAllById(saves.map { it.id.eventId })
            .filter { it.deletedAt == null }
            .associateBy { it.id }
        return saves.mapNotNull { save ->
            val event = events[save.id.eventId] ?: return@mapNotNull null
            toSummary(event, save.savedAt)
        }
    }

    fun save(userId: UUID, eventId: UUID): SavedEventSummaryDto {
        val event = eventRepository.findById(eventId).orElse(null)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        // Neither state is visible to a runner (/events excludes DRAFT via
        // EventService.LIST_DEFAULT_STATUSES), so both answer 404 rather than a
        // distinct code — saving either would write a row list() then filters
        // out, leaving an invisible orphan.
        if (!isRunnerVisible(event)) {
            throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        }
        val id = SavedEventId(userId = userId, eventId = eventId)
        val existing = savedEventRepository.findById(id).orElse(null)
        val savedAt = existing?.savedAt ?: savedEventRepository.save(SavedEvent(id)).savedAt
        return toSummary(event, savedAt)
    }

    fun unsave(userId: UUID, eventId: UUID): Boolean =
        savedEventRepository.deleteByUserIdAndEventId(userId, eventId) > 0

    fun merge(userId: UUID, incoming: Collection<UUID>): List<SavedEventSummaryDto> {
        val existing = savedEventRepository.findEventIdsByUserId(userId).toSet()
        val newIds = incoming.toSet() - existing
        if (newIds.isNotEmpty()) {
            // Same gate as save(), but merge skips instead of throwing: this is
            // a bulk login-time call, so one bad id must not abort the whole
            // merge (the FE fires it inside a Promise.all it can't partially
            // recover). Unknown ids were already dropped by findAllById.
            val knownNewIds = eventRepository.findAllById(newIds)
                .filter(::isRunnerVisible)
                .map { it.id }
                .toSet()
            knownNewIds.forEach { eventId ->
                savedEventRepository.save(SavedEvent(SavedEventId(userId, eventId)))
            }
        }
        return list(userId)
    }

    // An event a runner can actually reach. Soft-deleted rows are already
    // filtered by list(); DRAFT is admin-only pipeline state.
    private fun isRunnerVisible(event: Event): Boolean =
        event.deletedAt == null && event.status != EventStatus.DRAFT

    private fun toSummary(event: Event, savedAt: OffsetDateTime): SavedEventSummaryDto =
        SavedEventSummaryDto(
            id = event.id,
            slug = event.slug,
            name = event.name,
            date = event.date,
            state = EventDtoMapper.deriveAdminEventState(event),
            bannerUrl = eventDtoMapper.resolveBannerUrl(event),
            location = event.location,
            savedAt = savedAt,
        )
}
