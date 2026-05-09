package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Embeddable
import jakarta.persistence.EmbeddedId
import jakarta.persistence.Entity
import jakarta.persistence.Table
import java.io.Serializable
import java.time.OffsetDateTime
import java.util.UUID

@Embeddable
data class SavedEventId(
    @Column(name = "user_id", nullable = false)
    var userId: UUID,

    @Column(name = "event_id", nullable = false)
    var eventId: UUID,
) : Serializable

@Entity
@Table(name = "saved_events")
class SavedEvent(
    @EmbeddedId
    val id: SavedEventId,

    @Column(name = "saved_at", nullable = false)
    val savedAt: OffsetDateTime = OffsetDateTime.now(),
)
