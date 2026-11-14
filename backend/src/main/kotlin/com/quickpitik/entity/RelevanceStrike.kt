package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// RV-2.1 (P6): one confirmed case that counts toward the photographer's ladder (V106 relevance_strikes). No foreign key
// to the case or the event, so the count survives their deletion. Written only by service/relevance; RV-4.1 adds the writer.
@Entity
@Table(name = "relevance_strikes")
class RelevanceStrike(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),
    @Column(name = "photographer_id", nullable = false, updatable = false) val photographerId: UUID,
    @Column(name = "case_id", nullable = false, updatable = false) val caseId: UUID,
    @Column(name = "event_id", nullable = false, updatable = false) val eventId: UUID,
    @Column(name = "verdict", nullable = false, updatable = false, length = 12) val verdict: String,
    @Column(name = "confirmed_at", nullable = false, updatable = false) val confirmedAt: OffsetDateTime,
)
