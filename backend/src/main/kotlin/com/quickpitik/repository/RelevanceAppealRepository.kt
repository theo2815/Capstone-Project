package com.quickpitik.repository

import com.quickpitik.entity.RelevanceAppeal
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.Instant
import java.util.UUID

// RV-2.1: relevance_appeals (V106), written only by service/relevance. RV-3.2a: one appeal per case (unique case_id),
// one open per batch (uq_relevance_appeals_open).
interface RelevanceAppealRepository : JpaRepository<RelevanceAppeal, UUID> {
    fun findByCaseId(caseId: UUID): RelevanceAppeal?

    fun existsByBatchIdAndStatus(batchId: UUID, status: String): Boolean

    // RV-3.2 part b: the admin's evidence look and decision lock the appeal row.
    @Query(value = "SELECT * FROM relevance_appeals WHERE id = :id FOR UPDATE", nativeQuery = true)
    fun lockAppeal(@Param("id") id: UUID): RelevanceAppeal?

    // RV-3.2 part b: the admin's appeals list, newest first; :status is OPEN, WON, LOST or all (the service checks it).
    // The event and the people are left joins, so an appeal whose event was deleted stays listed.
    @Query(value = "SELECT $APPEAL_ROW $APPEAL_FROM WHERE (:status = 'all' OR a.status = :status) ORDER BY a.created_at DESC, a.id",
        countQuery = "SELECT count(*) $APPEAL_FROM WHERE (:status = 'all' OR a.status = :status)", nativeQuery = true)
    fun pageAppeals(@Param("status") status: String, pageable: Pageable): Page<RelevanceAppealRow>

    @Query(value = "SELECT $APPEAL_ROW $APPEAL_FROM WHERE a.id = :id", nativeQuery = true)
    fun appealRow(@Param("id") id: UUID): RelevanceAppealRow?
}

private const val APPEAL_ROW = """a.id AS "appealId", a.case_id AS "caseId", a.batch_id AS "batchId", br.event_id AS "eventId",
    e.name AS "eventName", u.name AS "photographerName", ps.handle AS "photographerHandle", c.verdict AS "verdict",
    c.verdict_note AS "verdictNote", c.action AS "action", c.stage AS "caseStage", a.status AS "status", a.reason AS "reason",
    a.created_at AS "createdAt", a.evidence_opened_at AS "evidenceOpenedAt", a.evidence_opened_by AS "evidenceOpenedBy",
    a.decided_at AS "decidedAt", a.decided_by AS "decidedBy", a.decision_note AS "decisionNote""""

private const val APPEAL_FROM = """FROM relevance_appeals a
    JOIN relevance_cases c ON c.id = a.case_id
    JOIN batch_relevance br ON br.id = a.batch_id
    LEFT JOIN events e ON e.id = br.event_id
    LEFT JOIN users u ON u.id = a.photographer_id
    LEFT JOIN photographer_settings ps ON ps.user_id = a.photographer_id"""

interface RelevanceAppealRow {
    val appealId: UUID
    val caseId: UUID
    val batchId: UUID
    val eventId: UUID
    val eventName: String?
    val photographerName: String?
    val photographerHandle: String?
    val verdict: String?
    val verdictNote: String?
    val action: String?
    val caseStage: String
    val status: String
    val reason: String
    val createdAt: Instant?
    val evidenceOpenedAt: Instant?
    val evidenceOpenedBy: UUID?
    val decidedAt: Instant?
    val decidedBy: UUID?
    val decisionNote: String?
}
