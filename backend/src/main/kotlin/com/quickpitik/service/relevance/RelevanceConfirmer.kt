package com.quickpitik.service.relevance

import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceStrikeRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

// RV-3.2 part b (A.7): confirming a case. It stamps confirmed_at and writes the case's one relevance_strikes row, in the
// caller's transaction; true only for the call that confirmed it, so a second call changes nothing. A lost appeal calls
// it now; RV-4.1's 14-day job reuses it and adds the ladder.
@Service
class RelevanceConfirmer(
    private val cases: RelevanceCaseRepository,
    private val strikes: RelevanceStrikeRepository,
) {
    @Transactional
    fun confirm(caseId: UUID, now: OffsetDateTime): Boolean {
        if (cases.stampConfirmed(caseId, now) != 1) return false
        strikes.insertForCase(caseId)
        return true
    }
}
