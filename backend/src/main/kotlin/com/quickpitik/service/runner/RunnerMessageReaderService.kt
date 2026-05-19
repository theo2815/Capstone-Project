package com.quickpitik.service.runner

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.MarkAllReadResponse
import com.quickpitik.dto.photographer.MessageRemovedResponse
import com.quickpitik.dto.runner.RunnerMessageDto
import com.quickpitik.entity.RunnerMessage
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.RunnerMessageRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

// Runner-facing inbox reader. Every action gates on runnerId (the
// authenticated user's id) so admin-side writes can never be mutated via
// this surface — see findByIdAndRunnerId on the repo for the IDOR guard.
//
// Soft-delete: `removed_at` is set on DELETE; the list endpoint filters
// it. Hard-delete is intentionally avoided so the admin-side decision-log
// chain (admin_decision_log.id ← runner_messages.source_decision_id)
// remains resolvable for audit views. Mirrors PhotographerMessageService.
@Service
@Transactional
class RunnerMessageReaderService(
    private val runnerMessageRepository: RunnerMessageRepository,
) {

    @Transactional(readOnly = true)
    fun list(runnerId: UUID): List<RunnerMessageDto> =
        runnerMessageRepository
            .findByRunnerIdAndRemovedAtIsNullOrderByCreatedAtDescIdAsc(runnerId)
            .map { it.toDto() }

    fun markRead(runnerId: UUID, messageId: UUID): RunnerMessageDto {
        val message = loadOwn(runnerId, messageId)
        if (message.readAt == null) {
            message.readAt = OffsetDateTime.now()
            runnerMessageRepository.save(message)
        }
        return message.toDto()
    }

    fun markAllRead(runnerId: UUID): MarkAllReadResponse {
        val updated = runnerMessageRepository.markAllReadForRunner(
            runnerId = runnerId,
            readAt = OffsetDateTime.now(),
        )
        return MarkAllReadResponse(markedRead = updated)
    }

    fun remove(runnerId: UUID, messageId: UUID): MessageRemovedResponse {
        val message = loadOwn(runnerId, messageId)
        if (message.removedAt == null) {
            message.removedAt = OffsetDateTime.now()
            runnerMessageRepository.save(message)
        }
        return MessageRemovedResponse(removed = true)
    }

    private fun loadOwn(runnerId: UUID, messageId: UUID): RunnerMessage =
        runnerMessageRepository
            .findByIdAndRunnerId(messageId, runnerId)
            .orElseThrow {
                NotFoundException(
                    code = ErrorCodes.NOT_FOUND,
                    message = "Message not found",
                )
            }

    private fun RunnerMessage.toDto(): RunnerMessageDto =
        RunnerMessageDto(
            id = id,
            kind = kindWire,
            title = title,
            body = body,
            orderId = orderId,
            sourceDecisionId = sourceDecisionId,
            createdAt = createdAt,
            readAt = readAt,
        )
}
