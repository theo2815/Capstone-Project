package com.quickpitik.repository

import com.quickpitik.entity.AiIndexJob
import com.quickpitik.entity.IndexJobStatus
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.UUID

interface AiIndexJobRepository : JpaRepository<AiIndexJob, UUID> {

    fun findByAiJobId(aiJobId: String): AiIndexJob?

    fun findByBatchId(batchId: UUID): List<AiIndexJob>

    // Poll/reconcile backstop: jobs still awaiting results past a grace window
    // (webhook lost, or webhooks disabled in dev). Backed by the partial index
    // idx_ai_index_jobs_open (migration V22).
    @Query(
        """
        SELECT j FROM AiIndexJob j
        WHERE j.status = :status
          AND j.submittedAt < :cutoff
        ORDER BY j.submittedAt ASC
        """,
    )
    fun findOpenSubmittedBefore(
        @Param("status") status: IndexJobStatus,
        @Param("cutoff") cutoff: OffsetDateTime,
        pageable: Pageable,
    ): List<AiIndexJob>
}
