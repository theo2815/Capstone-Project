package com.quickpitik.repository

import com.quickpitik.entity.AdminDecisionLog
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.UUID

interface AdminDecisionLogRepository : JpaRepository<AdminDecisionLog, UUID> {

    @Query(
        """
        SELECT a FROM AdminDecisionLog a
        WHERE a.targetUserId = :targetUserId
        ORDER BY a.decidedAt DESC, a.id ASC
        """,
    )
    fun findForUser(@Param("targetUserId") targetUserId: UUID): List<AdminDecisionLog>

    // Counts logical decisions in the window — collapses bulk groups (same
    // group_id) down to one logical decision per Q-A4. Native SQL because
    // JPQL has no clean COALESCE/cast for the group_id-or-id case.
    @Query(
        value = """
        SELECT COUNT(*) FROM (
            SELECT COALESCE(group_id::text, id::text) AS bucket
            FROM admin_decision_log
            WHERE decided_at >= :from AND decided_at < :to
            GROUP BY bucket
        ) t
        """,
        nativeQuery = true,
    )
    fun countLogicalDecisionsBetween(
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): Long

    // KPI trend per day — native SQL bucket on decided_at::date.
    @Query(
        value = """
        SELECT decided_at::date AS day,
               COUNT(DISTINCT COALESCE(group_id::text, id::text)) AS decision_count
        FROM admin_decision_log
        WHERE decided_at >= :from AND decided_at < :to
        GROUP BY day
        ORDER BY day
        """,
        nativeQuery = true,
    )
    fun decisionCountsByDayBetween(
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): List<Array<Any>>

    // Idempotent-bulk fast path (C-3): given a (admin_id, idempotency_key)
    // pair, return the group_id that the original bulk run minted so the
    // retry reuses it instead of generating a fresh UUID. LIMIT 1 because
    // every row in a single bulk run shares the same group_id by
    // construction. Returns null when no prior bulk run claimed this key.
    // Native SQL because JPQL has no clean LIMIT clause without a Pageable
    // overload — and we don't need pagination semantics here, just "first
    // (and only) groupId or null".
    @Query(
        value = """
        SELECT group_id FROM admin_decision_log
        WHERE admin_id = :adminId
          AND idempotency_key = :idempotencyKey
        LIMIT 1
        """,
        nativeQuery = true,
    )
    fun findGroupIdByAdminAndIdempotencyKey(
        @Param("adminId") adminId: UUID,
        @Param("idempotencyKey") idempotencyKey: String,
    ): UUID?

    // Per-row dedup gate (C-3): inside a bulk replay, skip the
    // logPayoutDecision + pushMessage writes when a row already exists for
    // (admin_id, idempotency_key, target_payout_id). Backed by the partial
    // unique index uq_admin_decision_log_admin_idem_target_payout — the
    // existence check is the cheap pre-flight; the index is the absolute
    // guarantee under any concurrent racing.
    fun existsByAdminIdAndIdempotencyKeyAndTargetPayoutId(
        adminId: UUID,
        idempotencyKey: String,
        targetPayoutId: String,
    ): Boolean
}
