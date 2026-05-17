package com.quickpitik.service.admin

import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.PhotographerMessage
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.websocket.PhotographerMessageCreatedEvent
import org.springframework.context.ApplicationEventPublisher
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

/**
 * Single writer for the admin audit trail + photographer-inbox per Q-A2 / Q-A4.
 *
 * Every admin write action (approve user, hold payout, resolve dispute, ...)
 * funnels through here so:
 *  1. The decision-log row is written in the same TX as the action.
 *  2. The photographer's inbox gets a message row (when the action affects
 *     a photographer) in the same TX, no out-of-band webhook needed.
 *
 * Bulk endpoints (POST /admin/payouts/bulk) generate one group_id UUID up
 * front and pass it on every per-target call so the dashboard kpi can
 * collapse the bulk decision back to one logical action.
 */
@Service
class AdminDecisionLogService(
    private val adminDecisionLogRepository: AdminDecisionLogRepository,
    private val photographerMessageRepository: PhotographerMessageRepository,
    private val eventPublisher: ApplicationEventPublisher,
) {

    @Transactional(propagation = Propagation.MANDATORY)
    fun logUserDecision(
        adminId: UUID,
        targetUserId: UUID,
        decision: String,
        reason: String? = null,
        meta: Map<String, Any?>? = null,
        groupId: UUID? = null,
    ): AdminDecisionLog =
        adminDecisionLogRepository.save(
            AdminDecisionLog(
                adminId = adminId,
                targetUserId = targetUserId,
                decision = decision,
                reason = reason,
                meta = meta,
                groupId = groupId,
            ),
        )

    @Transactional(propagation = Propagation.MANDATORY)
    fun logPayoutDecision(
        adminId: UUID,
        targetPayoutId: String,
        decision: String,
        reason: String? = null,
        meta: Map<String, Any?>? = null,
        groupId: UUID? = null,
        idempotencyKey: String? = null,
    ): AdminDecisionLog =
        adminDecisionLogRepository.save(
            AdminDecisionLog(
                adminId = adminId,
                targetPayoutId = targetPayoutId,
                decision = decision,
                reason = reason,
                meta = meta,
                groupId = groupId,
                idempotencyKey = idempotencyKey,
            ),
        )

    @Transactional(propagation = Propagation.MANDATORY)
    fun logDisputeDecision(
        adminId: UUID,
        targetDisputeId: UUID,
        decision: String,
        reason: String? = null,
        meta: Map<String, Any?>? = null,
        groupId: UUID? = null,
    ): AdminDecisionLog =
        adminDecisionLogRepository.save(
            AdminDecisionLog(
                adminId = adminId,
                targetDisputeId = targetDisputeId,
                decision = decision,
                reason = reason,
                meta = meta,
                groupId = groupId,
            ),
        )

    @Transactional(propagation = Propagation.MANDATORY)
    fun logEventDecision(
        adminId: UUID,
        targetEventId: UUID,
        decision: String,
        reason: String? = null,
        meta: Map<String, Any?>? = null,
    ): AdminDecisionLog =
        adminDecisionLogRepository.save(
            AdminDecisionLog(
                adminId = adminId,
                targetEventId = targetEventId,
                decision = decision,
                reason = reason,
                meta = meta,
            ),
        )

    @Transactional(propagation = Propagation.MANDATORY)
    fun pushMessage(
        photographerId: UUID,
        kind: PhotographerMessageKind,
        body: String,
        sourceAdminId: UUID? = null,
        sourceDecisionId: UUID? = null,
        /** Optional title — populated by ADMIN_MESSAGE (admin-supplied subject).
         *  For every other kind the FE derives the label from MESSAGE_KIND_LABEL,
         *  so callers leave this null. */
        title: String? = null,
    ): PhotographerMessage {
        val saved = photographerMessageRepository.save(
            PhotographerMessage(
                photographerId = photographerId,
                kindWire = kind.wire,
                title = title,
                body = body,
                sourceAdminId = sourceAdminId,
                sourceDecisionId = sourceDecisionId,
            ),
        )
        // Publish for the WS broadcaster — fires only AFTER_COMMIT so a
        // rollback of the surrounding admin TX drops the would-be push.
        // Payload mirrors PhotographerMessageDto so the FE merges directly
        // into useMyPhotographerMessagesStore without a refetch.
        eventPublisher.publishEvent(
            PhotographerMessageCreatedEvent(
                photographerId = photographerId,
                payload = mapOf(
                    "type" to "message.created",
                    "message" to mapOf(
                        "id" to saved.id.toString(),
                        "kind" to saved.kindWire,
                        "title" to saved.title,
                        "body" to saved.body,
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
