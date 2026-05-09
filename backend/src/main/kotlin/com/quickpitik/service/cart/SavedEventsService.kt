package com.quickpitik.service.cart

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.SavedEvent
import com.quickpitik.entity.SavedEventId
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.SavedEventRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional
class SavedEventsService(
    private val savedEventRepository: SavedEventRepository,
    private val eventRepository: EventRepository,
) {
    @Transactional(readOnly = true)
    fun list(userId: UUID): List<UUID> = savedEventRepository.findEventIdsByUserId(userId)

    fun save(userId: UUID, eventId: UUID): OffsetDateTime {
        if (!eventRepository.existsById(eventId)) {
            throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        }
        val id = SavedEventId(userId = userId, eventId = eventId)
        val existing = savedEventRepository.findById(id).orElse(null)
        if (existing != null) return existing.savedAt
        val saved = savedEventRepository.save(SavedEvent(id))
        return saved.savedAt
    }

    fun unsave(userId: UUID, eventId: UUID): Boolean =
        savedEventRepository.deleteByUserIdAndEventId(userId, eventId) > 0

    fun merge(userId: UUID, incoming: Collection<UUID>): List<UUID> {
        val existing = savedEventRepository.findEventIdsByUserId(userId).toSet()
        val newIds = incoming.toSet() - existing
        if (newIds.isNotEmpty()) {
            val knownNewIds = eventRepository.findAllById(newIds).map { it.id }.toSet()
            knownNewIds.forEach { eventId ->
                savedEventRepository.save(SavedEvent(SavedEventId(userId, eventId)))
            }
        }
        return savedEventRepository.findEventIdsByUserId(userId)
    }
}
