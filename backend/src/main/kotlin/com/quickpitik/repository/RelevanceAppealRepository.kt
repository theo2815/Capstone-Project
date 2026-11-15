package com.quickpitik.repository

import com.quickpitik.entity.RelevanceAppeal
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

// RV-2.1: relevance_appeals (V106), written only by service/relevance. RV-3.2a: one appeal per case (unique case_id),
// one open per batch (uq_relevance_appeals_open).
interface RelevanceAppealRepository : JpaRepository<RelevanceAppeal, UUID> {
    fun findByCaseId(caseId: UUID): RelevanceAppeal?

    fun existsByBatchIdAndStatus(batchId: UUID, status: String): Boolean
}
