package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// One ai-api job inside a batch: either the FACE mega (enroll) or the BIB mega
// (recognize). aiJobId is the ai-api job UUID and the idempotency anchor — the
// webhook receiver and the poll backstop both key off it (UNIQUE), so a
// duplicate delivery on an already-terminal job is a no-op. See migration V22.
@Entity
@Table(name = "ai_index_jobs")
class AiIndexJob(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "batch_id", nullable = false)
    val batchId: UUID,

    @Enumerated(EnumType.STRING)
    @Column(name = "kind", nullable = false, length = 8)
    val kind: IndexJobKind,

    @Column(name = "ai_job_id", nullable = false, length = 64)
    val aiJobId: String,

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 16)
    var status: IndexJobStatus = IndexJobStatus.SUBMITTED,

    @Column(name = "submitted_at", nullable = false)
    val submittedAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "completed_at")
    var completedAt: OffsetDateTime? = null,

    @Column(name = "error", columnDefinition = "TEXT")
    var error: String? = null,
)

enum class IndexJobKind {
    FACE,
    BIB,
}

enum class IndexJobStatus {
    SUBMITTED,
    COMPLETED,
    FAILED,
}
