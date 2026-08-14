package com.quickpitik.repository

import com.quickpitik.entity.RunnerMessage
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID

interface RunnerMessageRepository : JpaRepository<RunnerMessage, UUID> {

    // List for /me/runner/messages — filters soft-deleted rows. Paged; mirrors
    // PhotographerMessageRepository exactly.
    fun findByRunnerIdAndRemovedAtIsNullOrderByCreatedAtDescIdAsc(
        runnerId: UUID,
        pageable: Pageable,
    ): List<RunnerMessage>

    // Tenant-safe single-row fetch — guards against IDOR by matching runner_id.
    fun findByIdAndRunnerId(
        id: UUID,
        runnerId: UUID,
    ): Optional<RunnerMessage>

    @Modifying
    @Query(
        "UPDATE RunnerMessage m SET m.readAt = :readAt " +
            "WHERE m.runnerId = :runnerId AND m.readAt IS NULL AND m.removedAt IS NULL",
    )
    fun markAllReadForRunner(
        @Param("runnerId") runnerId: UUID,
        @Param("readAt") readAt: OffsetDateTime,
    ): Int
}
