package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceMode
import com.quickpitik.entity.CaseStage
import com.quickpitik.entity.CaseVerdict
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceStrikeRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.admin.AdminFlagService
import com.quickpitik.service.admin.SystemFlagService
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.util.UUID

// RV-4.2 (part 2, A.8, O1): the automatic payout hold for copied photos. It reuses the only hold there is, an open
// relevance flag on the photographer with payout_hold_at set, so it is account-wide (the one exception to "per batch").
// The case's own note (V2-T14) and its COPIED alert already tell the photographer and the admins, so placing it sends
// neither. It ends only through AdminFlagService, which sends the existing "Payout hold lifted" note.
// Lock order, place and end alike: the case row, then the photographer's user row, then the flag. The confirm step and
// a lost appeal take the case, then the user row (RV-4.1); AdminFlagService takes the user row, then the flag.
@Service
class RelevanceHoldService(
    private val cases: RelevanceCaseRepository,
    private val strikes: RelevanceStrikeRepository,
    private val systemFlags: SystemFlagService,
    private val adminFlags: AdminFlagService,
    private val decisionLog: AdminDecisionLogService,
    private val switch: RelevanceSwitch,
    private val transactions: TransactionTemplate,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // In its own transaction, after the apply step commits. Only a current Copied case applied in auto-hide (consent
    // included) holds payouts, once: a case that already names its hold flag changes nothing.
    fun place(caseId: UUID) {
        if (switch.mode != RelevanceMode.AUTO_HIDE) return
        transactions.executeWithoutResult {
            cases.lockCase(caseId) ?: return@executeWithoutResult
            val case = cases.findById(caseId).orElseThrow()
            if (case.stage != CaseStage.APPLIED.name || case.appliedMode != RelevanceMode.AUTO_HIDE.name ||
                case.verdict != CaseVerdict.COPIED.name || case.holdFlagId != null) return@executeWithoutResult
            val photographer = strikes.lockPhotographerOf(caseId)!!
            val flagId = systemFlags.holdPayouts(photographer, "Photo Relevance case $caseId: copied photos")
            cases.setHoldFlag(caseId, flagId)
            decisionLog.logAutomaticUserDecision(photographer, "payout_hold_auto",
                meta = mapOf("caseId" to caseId.toString(), "flagId" to flagId.toString()))
            log.info("Relevance payout hold placed case={} flag={}", caseId, flagId)
        }
    }

    // In the caller's transaction, which holds the case row. Restore (an admin restore, a won appeal) dismisses the flag;
    // an admin's "End hold" resolves it. False when the case holds no payouts.
    fun end(caseId: UUID, adminId: UUID, restore: Boolean): Boolean {
        val flagId = cases.liveHoldFlag(caseId) ?: return false
        val note = "Photo Relevance case $caseId"
        if (restore) adminFlags.dismiss(adminId, flagId, note) else adminFlags.resolve(adminId, flagId, note)
        return true
    }
}
