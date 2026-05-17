package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.MarkAllReadResponse
import com.quickpitik.dto.photographer.MessageRemovedResponse
import com.quickpitik.dto.photographer.PhotographerMessageDto
import com.quickpitik.entity.PhotographerMessage
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.PhotographerMessageRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

// Photographer-facing inbox reader. Every action gates on photographerId
// (the authenticated user's id) so admin-side mutations can never be
// flipped/removed via this surface — see findByIdAndPhotographerId on the
// repo for the IDOR guard.
//
// Soft-delete: `removed_at` is set on DELETE; the list endpoint filters
// it. Hard-delete is intentionally avoided so the admin-side decision-log
// chain (admin_decision_log.id ← photographer_messages.source_decision_id)
// remains resolvable for audit views.
@Service
@Transactional
class PhotographerMessageService(
    private val photographerMessageRepository: PhotographerMessageRepository,
) {

    @Transactional(readOnly = true)
    fun list(photographerId: UUID): List<PhotographerMessageDto> =
        photographerMessageRepository
            .findByPhotographerIdAndRemovedAtIsNullOrderByCreatedAtDescIdAsc(photographerId)
            .map { it.toDto() }

    fun markRead(photographerId: UUID, messageId: UUID): PhotographerMessageDto {
        val message = loadOwn(photographerId, messageId)
        if (message.readAt == null) {
            message.readAt = OffsetDateTime.now()
            photographerMessageRepository.save(message)
        }
        return message.toDto()
    }

    fun markAllRead(photographerId: UUID): MarkAllReadResponse {
        val updated = photographerMessageRepository.markAllReadForPhotographer(
            photographerId = photographerId,
            readAt = OffsetDateTime.now(),
        )
        return MarkAllReadResponse(markedRead = updated)
    }

    // Soft-delete. Idempotent: a second call on an already-removed row
    // returns removed=true without re-touching the timestamp.
    fun remove(photographerId: UUID, messageId: UUID): MessageRemovedResponse {
        val message = loadOwn(photographerId, messageId)
        if (message.removedAt == null) {
            message.removedAt = OffsetDateTime.now()
            photographerMessageRepository.save(message)
        }
        return MessageRemovedResponse(removed = true)
    }

    private fun loadOwn(photographerId: UUID, messageId: UUID): PhotographerMessage =
        photographerMessageRepository
            .findByIdAndPhotographerId(messageId, photographerId)
            .orElseThrow {
                NotFoundException(
                    code = ErrorCodes.NOT_FOUND,
                    message = "Message not found",
                )
            }

    private fun PhotographerMessage.toDto(): PhotographerMessageDto =
        PhotographerMessageDto(
            id = id,
            kind = kindWire,
            title = title,
            body = body,
            sourceDecisionId = sourceDecisionId,
            createdAt = createdAt,
            readAt = readAt,
        )
}
