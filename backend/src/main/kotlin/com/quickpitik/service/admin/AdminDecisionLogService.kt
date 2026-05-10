package com.quickpitik.service.admin

import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.PhotographerMessage
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.PhotographerMessageRepository
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
    ): AdminDecisionLog =
        adminDecisionLogRepository.save(
            AdminDecisionLog(
                adminId = adminId,
                targetPayoutId = targetPayoutId,
                decision = decision,
                reason = reason,
                meta = meta,
                groupId = groupId,
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
    ): PhotographerMessage =
        photographerMessageRepository.save(
            PhotographerMessage(
                photographerId = photographerId,
                kindWire = kind.wire,
                body = body,
                sourceAdminId = sourceAdminId,
                sourceDecisionId = sourceDecisionId,
            ),
        )
}
