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

    @Modifying
    @Query(value = """
        INSERT INTO runner_messages (id, runner_id, kind, body, order_id, photo_removal_request_id, notice_key, created_at)
        VALUES (:id, :runnerId, 'photo_removed', :body, :orderId, :requestId, :noticeKey, :createdAt)
        ON CONFLICT (notice_key) WHERE notice_key IS NOT NULL DO NOTHING
    """, nativeQuery = true)
    fun insertRemovalNotice(id: UUID, runnerId: UUID, orderId: UUID, requestId: UUID, noticeKey: String,
        body: String, createdAt: OffsetDateTime): Int

    // List for /me/runner/messages — filters soft-deleted rows. Paged; mirrors
    // PhotographerMessageRepository exactly.
    fun findByRunnerIdAndRemovedAtIsNullOrderByCreatedAtDescIdAsc(
        runnerId: UUID,
        pageable: Pageable,
    ): List<RunnerMessage>

    // Total un-removed messages — surfaced via the X-Total-Count header so the
    // web inbox knows the true count behind its capped page.
    fun countByRunnerIdAndRemovedAtIsNull(runnerId: UUID): Long

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
