package com.quickpitik.service.runner

import com.quickpitik.entity.RunnerMessage
import com.quickpitik.entity.RunnerMessageKind
import com.quickpitik.repository.RunnerMessageRepository
import com.quickpitik.websocket.RunnerMessageCreatedEvent
import org.springframework.context.ApplicationEventPublisher
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

/**
 * Single writer for the runner-side inbox. Mirrors
 * AdminDecisionLogService.pushMessage but lands in `runner_messages`
 * instead of `photographer_messages`.
 *
 * Every admin action that targets a runner (dispute resolve / deny /
 * escalate, account suspend / unsuspend, admin DM) funnels through here so
 * the inbox row is written in the same TX as the decision-log entry. The
 * `@Propagation.MANDATORY` means callers MUST already be in a transaction —
 * matches AdminDecisionLogService.
 *
 * Publishes RunnerMessageCreatedEvent which NotificationBroadcaster
 * forwards to the /ws/me/runner/notifications channel on AFTER_COMMIT.
 */
@Service
class RunnerMessagesService(
    private val runnerMessageRepository: RunnerMessageRepository,
    private val eventPublisher: ApplicationEventPublisher,
) {
    @Transactional(propagation = Propagation.MANDATORY)
    fun pushMessage(
        runnerId: UUID,
        kind: RunnerMessageKind,
        body: String,
        sourceAdminId: UUID? = null,
        sourceDecisionId: UUID? = null,
        orderId: UUID? = null,
        /** Optional title — populated by ADMIN_MESSAGE (admin-supplied
         *  subject). For every other kind the FE derives the label from
         *  the kind, so callers leave this null. */
        title: String? = null,
    ): RunnerMessage {
        val saved = runnerMessageRepository.save(
            RunnerMessage(
                runnerId = runnerId,
                kindWire = kind.wire,
                title = title,
                body = body,
                sourceAdminId = sourceAdminId,
                sourceDecisionId = sourceDecisionId,
                orderId = orderId,
            ),
        )
        // Fires only AFTER_COMMIT so a rollback of the surrounding admin
        // TX drops the would-be push. Payload mirrors the FE RunnerMessage
        // shape so the bell merges directly without a refetch.
        eventPublisher.publishEvent(
            RunnerMessageCreatedEvent(
                runnerId = runnerId,
                payload = mapOf(
                    "type" to "message.created",
                    "message" to mapOf(
                        "id" to saved.id.toString(),
                        "kind" to saved.kindWire,
                        "title" to saved.title,
                        "body" to saved.body,
                        "orderId" to saved.orderId?.toString(),
                        "sourceDecisionId" to saved.sourceDecisionId?.toString(),
                        "createdAt" to saved.createdAt.toString(),
                        "readAt" to saved.readAt?.toString(),
                    ),
                ),
            ),
        )
        return saved
    }
}
