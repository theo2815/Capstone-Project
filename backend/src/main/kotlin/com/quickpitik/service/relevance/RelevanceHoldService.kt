package com.quickpitik.service.relevance

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.entity.CaseStage
import com.quickpitik.entity.CaseVerdict
import com.quickpitik.entity.RelevanceCase
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceStrikeRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.admin.AdminFlagService
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.exception.ApiException
import org.slf4j.LoggerFactory
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.util.UUID

// RV-4.2 (part 2, A.8, O1): the automatic payout hold for copied photos. It uses the only hold there is, an open relevance
// flag on the photographer with payout_hold_at set, so it is account-wide (the one exception to "per batch"). Each copied
// case holds on a flag of its own, so each case's hold ends on its own and payouts stay held until the last one ends.
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
    private val admin: AdminProperties,
    private val users: UserRepository,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // In its own transaction, after the apply step commits. Only a current Copied case applied in auto-hide (consent
    // included) holds payouts, once: a case that already names its hold flag changes nothing.
    fun place(caseId: UUID) {
        if (switch.mode != RelevanceMode.AUTO_HIDE) return
        transactions.executeWithoutResult {
            cases.lockCase(caseId) ?: return@executeWithoutResult
            val case = cases.findById(caseId).orElseThrow()
            // RV-4.5a: an admin's hide (of a no-consent case) never holds payouts by itself; the admin's hold-payouts does.
            if (case.stage != CaseStage.APPLIED.name || case.appliedMode != RelevanceMode.AUTO_HIDE.name ||
                case.verdict != CaseVerdict.COPIED.name || case.holdFlagId != null || case.verdictNote == ADMIN_HIDE) return@executeWithoutResult
            val photographer = strikes.lockPhotographerOf(caseId)!!
            // RV-L7 part a: read under the user row lock, so a withdrawal recorded after the case was picked still stops it.
            if (users.relevanceConsentWithdrawnAt(photographer) != null) return@executeWithoutResult
            val flagId = systemFlags.holdPayouts(photographer, "Photo Relevance case $caseId: copied photos")
            cases.setHoldFlag(caseId, flagId)
            decisionLog.logAutomaticUserDecision(photographer, "payout_hold_auto",
                meta = mapOf("caseId" to caseId.toString(), "flagId" to flagId.toString()))
            log.info("Relevance payout hold placed case={} flag={}", caseId, flagId)
        }
    }

    // RV-4.5a: an admin holds the payouts of a current no-consent Copied case (record-only with NO_CONSENT, or since hidden
    // by an admin) that holds none yet, in the caller's transaction, which holds the case row (then the user row, then
    // the flag). The case gets a flag of its own; the hold goes on through AdminFlagService.confirmPayoutHold, so it is
    // the admin's hold with its approved "Payouts on hold" note, and End hold, a restore or a won appeal end it as before.
    fun placeByAdmin(case: RelevanceCase, adminId: UUID): Boolean {
        val noConsent = (case.appliedMode == RelevanceMode.RECORD_ONLY.name && case.verdictNote == NO_CONSENT) ||
            (case.appliedMode == RelevanceMode.AUTO_HIDE.name && case.verdictNote == ADMIN_HIDE)
        if (case.stage != CaseStage.APPLIED.name || case.verdict != CaseVerdict.COPIED.name || case.holdFlagId != null || !noConsent) return false
        val photographer = strikes.lockPhotographerOf(case.id)!!
        val flagId = systemFlags.openCaseFlag(photographer, "Photo Relevance case ${case.id}: copied photos")
        adminFlags.confirmPayoutHold(adminId, flagId)
        cases.setHoldFlag(case.id, flagId)
        return true
    }

    // In the caller's transaction, which holds the case row. Restore (an admin restore, a won appeal) dismisses the case's
    // flag; an admin's "End hold" resolves it. False when the case holds no payouts. With the flags switch off a flag
    // cannot be closed (and the hold would bind again when it is back on), so the caller's action is refused.
    fun end(caseId: UUID, adminId: UUID, restore: Boolean): Boolean {
        val flagId = cases.liveHoldFlag(caseId) ?: return false
        if (!admin.flagsEnabled) {
            throw ApiException(HttpStatus.FORBIDDEN, ErrorCodes.FLAGS_DISABLED,
                "The admin flags switch is off, so this case's payout hold cannot be ended. Turn ADMIN_FLAGS_ENABLED on first.")
        }
        val note = "Photo Relevance case $caseId"
        if (restore) adminFlags.dismiss(adminId, flagId, note) else adminFlags.resolve(adminId, flagId, note)
        return true
    }

    private companion object {
        const val ADMIN_HIDE = "ADMIN_HIDE"
        const val NO_CONSENT = "NO_CONSENT"
    }
}
