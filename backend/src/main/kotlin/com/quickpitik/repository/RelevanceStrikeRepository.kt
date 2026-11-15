package com.quickpitik.repository

import com.quickpitik.entity.RelevanceStrike
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

// RV-2.1: relevance_strikes (V106). RV-3.2 part b writes the row of a confirmed case; RV-4.1 adds the ladder's reads.
interface RelevanceStrikeRepository : JpaRepository<RelevanceStrike, UUID> {

    // The one strike of a case just confirmed (stampConfirmed), copied from the case; a second call adds nothing.
    @Modifying
    @Query(value = """
        INSERT INTO relevance_strikes (photographer_id, case_id, event_id, verdict, confirmed_at)
        SELECT br.photographer_id, c.id, br.event_id, c.verdict, c.confirmed_at
        FROM relevance_cases c JOIN batch_relevance br ON br.id = c.batch_id
        WHERE c.id = :caseId AND c.confirmed_at IS NOT NULL
        ON CONFLICT (case_id) DO NOTHING
    """, nativeQuery = true)
    fun insertForCase(@Param("caseId") caseId: UUID): Int
}
