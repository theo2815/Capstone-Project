package com.quickpitik.repository

import com.quickpitik.entity.RelevanceAlert
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.Instant
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

    // RV-1.10: the admin page's alert list, newest first; :unseenOnly leaves out the alerts an admin marked seen.
    @Query(value = "SELECT $ALERT_ROW $ALERT_FROM WHERE (:unseenOnly = false OR a.seen_at IS NULL) ORDER BY a.created_at DESC, a.id",
        countQuery = "SELECT count(*) $ALERT_FROM WHERE (:unseenOnly = false OR a.seen_at IS NULL)", nativeQuery = true)
    fun pageAlerts(@Param("unseenOnly") unseenOnly: Boolean, pageable: Pageable): Page<RelevanceAlertRow>

    @Query(value = "SELECT $ALERT_ROW $ALERT_FROM WHERE a.batch_id = :batchId ORDER BY a.created_at DESC, a.id", nativeQuery = true)
    fun alertsForBatch(@Param("batchId") batchId: UUID): List<RelevanceAlertRow>

    @Query(value = "SELECT $ALERT_ROW $ALERT_FROM WHERE a.id = :id", nativeQuery = true)
    fun alertRow(@Param("id") id: UUID): RelevanceAlertRow?

    // The first admin to mark an alert seen is the one stored: 1 when this call stamped it, 0 when it was already seen.
    @Modifying
    @Query(value = "UPDATE relevance_alerts SET seen_at = :now, seen_by = :adminId WHERE id = :id AND seen_at IS NULL",
        nativeQuery = true)
    fun markSeen(@Param("id") alertId: UUID, @Param("adminId") adminId: UUID, @Param("now") now: OffsetDateTime): Int

    // RV-2.5a: "keep hidden" marks every unseen alert of the case seen, keeping an earlier admin's stamp.
    @Modifying
    @Query(value = "UPDATE relevance_alerts SET seen_at = :now, seen_by = :adminId WHERE case_id = :caseId AND seen_at IS NULL",
        nativeQuery = true)
    fun markCaseSeen(@Param("caseId") caseId: UUID, @Param("adminId") adminId: UUID, @Param("now") now: OffsetDateTime): Int
}

private const val ALERT_ROW = """a.id AS "alertId", a.case_id AS "caseId", br.event_id AS "eventId", e.name AS "eventName",
    ps.handle AS "photographerHandle", u.name AS "photographerName", a.kind AS "kind", a.would_have AS "wouldHave",
    a.photos AS "photos", a.created_at AS "createdAt", a.seen_at AS "seenAt""""

private const val ALERT_FROM = """FROM relevance_alerts a
    JOIN batch_relevance br ON br.id = a.batch_id
    JOIN events e ON e.id = br.event_id
    JOIN users u ON u.id = br.photographer_id
    LEFT JOIN photographer_settings ps ON ps.user_id = br.photographer_id"""

interface RelevanceAlertRow {
    val alertId: UUID
    val caseId: UUID?
    val eventId: UUID
    val eventName: String
    val photographerHandle: String?
    val photographerName: String
    val kind: String
    val wouldHave: Boolean
    val photos: Int?
    val createdAt: Instant?
    val seenAt: Instant?
}
