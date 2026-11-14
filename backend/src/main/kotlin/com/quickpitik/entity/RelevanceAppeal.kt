package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// RV-2.1: the values the status CHECK of relevance_appeals allows (V106). The column stays a string.
enum class AppealStatus { OPEN, WON, LOST }

// RV-2.1 (G11): a photographer's appeal of one case (V106 relevance_appeals). One per case, and one OPEN per batch
// (unique indexes). Written only by service/relevance; RV-3.2 adds the writer.
@Entity
@Table(name = "relevance_appeals")
class RelevanceAppeal(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),
    @Column(name = "case_id", nullable = false, updatable = false) val caseId: UUID,
    @Column(name = "batch_id", nullable = false, updatable = false) val batchId: UUID,
    @Column(name = "photographer_id", nullable = false, updatable = false) val photographerId: UUID,
    @Column(name = "reason", nullable = false, updatable = false, columnDefinition = "TEXT") val reason: String,
    @Column(name = "status", nullable = false, length = 8) var status: String = AppealStatus.OPEN.name,
    @Column(name = "created_at", insertable = false, updatable = false) val createdAt: OffsetDateTime? = null,
    @Column(name = "evidence_opened_at") var evidenceOpenedAt: OffsetDateTime? = null,
    @Column(name = "evidence_opened_by") var evidenceOpenedBy: UUID? = null,
    @Column(name = "decided_at") var decidedAt: OffsetDateTime? = null,
    @Column(name = "decided_by") var decidedBy: UUID? = null,
    @Column(name = "decision_note", columnDefinition = "TEXT") var decisionNote: String? = null,
)
