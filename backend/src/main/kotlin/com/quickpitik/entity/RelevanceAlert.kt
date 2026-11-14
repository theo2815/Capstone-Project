package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// RV-1.9: the values the kind CHECK of relevance_alerts allows (V105). The column stays a string.
enum class AlertKind { BIG_HIDE, COPIED, APPEAL, THIRD_CASE }

// RV-1.9 (plan 5.11, R12): one stored admin alert (V105 relevance_alerts), written only by service/relevance through
// RelevanceAlertRepository.raise. dedupe_key is unique, so a retry or a second server copy cannot raise it twice.
@Entity
@Table(name = "relevance_alerts")
class RelevanceAlert(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),
    @Column(name = "batch_id", nullable = false, updatable = false) val batchId: UUID,
    @Column(name = "case_id") val caseId: UUID? = null,
    @Column(name = "kind", nullable = false, length = 12) val kind: String,
    @Column(name = "dedupe_key", nullable = false, length = 120) val dedupeKey: String,
    @Column(name = "would_have", nullable = false) val wouldHave: Boolean,
    @Column(name = "photos") val photos: Int? = null,
    @Column(name = "created_at", insertable = false, updatable = false) val createdAt: OffsetDateTime? = null,
    @Column(name = "seen_at") var seenAt: OffsetDateTime? = null,
    @Column(name = "seen_by") var seenBy: UUID? = null,
)
