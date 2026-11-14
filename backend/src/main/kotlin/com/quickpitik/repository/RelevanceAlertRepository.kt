package com.quickpitik.repository

import com.quickpitik.entity.RelevanceAlert
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.UUID

// RV-1.9: relevance_alerts, written only by service/relevance. RV-1.10 adds the page queries.
interface RelevanceAlertRepository : JpaRepository<RelevanceAlert, UUID> {

    // One alert per dedupe key (plan 5.11): returns 1 when this call stored it, 0 when it was already there.
    @Modifying
    @Query(value = """
        INSERT INTO relevance_alerts (id, batch_id, case_id, kind, dedupe_key, would_have, photos, created_at)
        VALUES (:id, :batchId, :caseId, :kind, :dedupeKey, :wouldHave, :photos, :now)
        ON CONFLICT (dedupe_key) DO NOTHING
    """, nativeQuery = true)
    fun raise(
        @Param("id") id: UUID,
        @Param("batchId") batchId: UUID,
        @Param("caseId") caseId: UUID,
        @Param("kind") kind: String,
        @Param("dedupeKey") dedupeKey: String,
        @Param("wouldHave") wouldHave: Boolean,
        @Param("photos") photos: Int,
        @Param("now") now: OffsetDateTime,
    ): Int
}
