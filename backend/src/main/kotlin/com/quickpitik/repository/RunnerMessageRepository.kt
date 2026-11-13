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

    // RD-9: one status notice per runner deletion request and status (the notice key holds both).
    @Modifying
    @Query(value = """
        INSERT INTO runner_messages (id, runner_id, kind, title, body, photo_removal_request_id, notice_key, created_at)
        VALUES (:id, :runnerId, 'photo_removal_update', :title, :body, :requestId, :noticeKey, :createdAt)
        ON CONFLICT (notice_key) WHERE notice_key IS NOT NULL DO NOTHING
    """, nativeQuery = true)
    fun insertKeyedNotice(id: UUID, runnerId: UUID, requestId: UUID, noticeKey: String, title: String, body: String,
        createdAt: OffsetDateTime): Int

    // RD-9 email send claim (columns from V99, as PhotographerMessageRepository). Native: the entity does not map them.
    @Modifying
    @Query(value = "UPDATE runner_messages SET email_sent_at = :now WHERE id = :id AND email_sent_at IS NULL", nativeQuery = true)
    fun claimEmailSend(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int

    @Modifying
    @Query(value = "UPDATE runner_messages SET email_delivered_at = :at WHERE id = :id AND email_delivered_at IS NULL", nativeQuery = true)
    fun stampEmailDelivered(@Param("id") id: UUID, @Param("at") at: OffsetDateTime): Int

    // Status notices claimed before the cutoff and never delivered, from the last 24 hours.
    @Query(
        value = """SELECT id FROM runner_messages
            WHERE kind = 'photo_removal_update'
              AND email_sent_at < :cutoff AND email_delivered_at IS NULL AND created_at >= :since
            ORDER BY email_sent_at LIMIT :limit""",
        nativeQuery = true,
    )
    fun findStaleEmailClaimIds(@Param("cutoff") cutoff: OffsetDateTime, @Param("since") since: OffsetDateTime, @Param("limit") limit: Int): List<UUID>

    // Match the old timestamp and missing delivery stamp, so a second instance's fresh claim is not released.
    @Modifying
    @Query(
        value = "UPDATE runner_messages SET email_sent_at = NULL WHERE id = :id AND email_sent_at < :cutoff AND email_delivered_at IS NULL",
        nativeQuery = true,
    )
    fun releaseStaleEmailSend(@Param("id") id: UUID, @Param("cutoff") cutoff: OffsetDateTime): Int

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
