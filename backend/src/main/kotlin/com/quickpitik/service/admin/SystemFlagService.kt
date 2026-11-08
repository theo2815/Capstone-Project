package com.quickpitik.service.admin

import com.quickpitik.entity.Flag
import com.quickpitik.entity.FlagStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.repository.FlagRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

// The one writer for findings the backend raises itself (NULL reporter) and for
// user reports (reporterId set). Rows land in the existing admin flag queue.
//
// Transactions: REQUIRED joins the caller's transaction. A caller that throws
// after raise (a refused save) must call it with no transaction open, or the
// flag rolls back with the refusal.
//
// shortcut: the open-flag check is read-then-insert, so two simultaneous raises of
// the same finding can write two rows; add a partial unique index if admins see doubles.
@Service
class SystemFlagService(private val flagRepository: FlagRepository) {

    /** Writes an open flag unless the same (kind, target, reason, reporter) is still open or escalated. */
    @Transactional
    fun raise(
        kind: FlagTargetKind,
        targetId: UUID,
        reason: SystemFlagReason,
        note: String? = null,
        reporterId: UUID? = null,
    ): Boolean {
        if (openFlags(kind, targetId, reason).any { it.reporterId == reporterId }) return false
        flagRepository.save(
            Flag(
                targetKindWire = kind.wire,
                targetId = targetId,
                reporterId = reporterId,
                reason = reason.wire,
                note = note.orEmpty().take(NOTE_MAX),
            ),
        )
        return true
    }

    /** Adds a line to the open system flag (NULL reporter) for this finding; false when none is open. */
    @Transactional
    fun appendNote(kind: FlagTargetKind, targetId: UUID, reason: SystemFlagReason, note: String): Boolean {
        val flag = openFlags(kind, targetId, reason).firstOrNull { it.reporterId == null } ?: return false
        flag.note = (if (flag.note.isEmpty()) note else flag.note + "\n" + note).takeLast(NOTE_MAX)
        flagRepository.save(flag)
        return true
    }

    private fun openFlags(kind: FlagTargetKind, targetId: UUID, reason: SystemFlagReason): List<Flag> =
        flagRepository.findByTargetKindWireAndTargetIdAndReasonAndStatusWireIn(
            kind.wire, targetId, reason.wire, OPEN_STATUSES,
        )

    private companion object {
        const val NOTE_MAX = 2000
        val OPEN_STATUSES = listOf(FlagStatus.OPEN.wire, FlagStatus.ESCALATED.wire)
    }
}
