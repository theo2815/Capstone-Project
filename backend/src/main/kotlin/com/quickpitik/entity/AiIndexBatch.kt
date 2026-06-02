package com.quickpitik.entity

import jakarta.persistence.CollectionTable
import jakarta.persistence.Column
import jakarta.persistence.ElementCollection
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.FetchType
import jakarta.persistence.Id
import jakarta.persistence.JoinColumn
import jakarta.persistence.OrderColumn
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// A bulk-indexing batch (Phase C): one event's slice of PENDING photos submitted
// to ai-api as a FACE mega job + a BIB mega job. photoIds are ordered so ai-api's
// positional result index maps back to the right photo (ai-api also echoes a
// per-image ref = the photo id). The batch finalizes only once BOTH its jobs are
// terminal. See migration V22.
@Entity
@Table(name = "ai_index_batches")
class AiIndexBatch(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "event_id", nullable = false)
    val eventId: UUID,

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 16)
    var status: BatchStatus = BatchStatus.SUBMITTED,

    @Column(name = "created_at", nullable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "finalized_at")
    var finalizedAt: OffsetDateTime? = null,

    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(
        name = "ai_index_batch_photos",
        joinColumns = [JoinColumn(name = "batch_id")],
    )
    @OrderColumn(name = "idx")
    @Column(name = "photo_id", nullable = false)
    val photoIds: MutableList<UUID> = mutableListOf(),
)

enum class BatchStatus {
    SUBMITTED,
    FINALIZED,
    FAILED,
}
