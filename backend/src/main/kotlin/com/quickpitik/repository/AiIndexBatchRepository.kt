package com.quickpitik.repository

import com.quickpitik.entity.AiIndexBatch
import jakarta.persistence.LockModeType
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface AiIndexBatchRepository : JpaRepository<AiIndexBatch, UUID> {

    // Pessimistic lock so the FACE and BIB ingests of the same batch serialize:
    // whichever runs second sees the first's committed job status and is the one
    // that finalizes. Without the lock a webhook-driven concurrent pair could
    // both observe the sibling as still SUBMITTED and neither would finalize.
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT b FROM AiIndexBatch b WHERE b.id = :id")
    fun findByIdForUpdate(@Param("id") id: UUID): AiIndexBatch?
}
