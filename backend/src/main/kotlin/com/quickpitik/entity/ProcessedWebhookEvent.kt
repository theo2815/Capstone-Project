package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// One row per processed provider webhook event. The (provider, event_id) UNIQUE
// is the dedupe: insert-first claims the event before processing, so a redelivery
// conflicts and is skipped. Surrogate id keeps JPA simple; the UNIQUE does the work.
@Entity
@Table(name = "processed_webhook_events")
class ProcessedWebhookEvent(
    @Id
    @Column(name = "id", nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "provider", nullable = false, length = 40, updatable = false)
    val provider: String,

    @Column(name = "event_id", nullable = false, length = 120, updatable = false)
    val eventId: String,

    @Column(name = "received_at", nullable = false, updatable = false)
    val receivedAt: OffsetDateTime = OffsetDateTime.now(),
)
