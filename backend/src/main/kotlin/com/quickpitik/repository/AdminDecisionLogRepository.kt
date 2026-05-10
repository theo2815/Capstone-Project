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
}
