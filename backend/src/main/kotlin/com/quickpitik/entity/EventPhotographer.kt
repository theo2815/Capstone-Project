package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Embeddable
import jakarta.persistence.EmbeddedId
import jakarta.persistence.Entity
import jakarta.persistence.Table
import java.io.Serializable
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

@Embeddable
data class EventPhotographerId(
    @Column(name = "event_id", nullable = false)
    val eventId: UUID = UUID.randomUUID(),

    @Column(name = "photographer_id", nullable = false)
    val photographerId: UUID = UUID.randomUUID(),
) : Serializable

@Entity
@Table(name = "event_photographer")
class EventPhotographer(
    @EmbeddedId
    val id: EventPhotographerId,

    @Column(name = "joined_at", nullable = false, updatable = false)
    val joinedAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "photo_count", nullable = false)
    var photoCount: Int = 0,

    @Column(name = "sales_count", nullable = false)
    var salesCount: Int = 0,

    @Column(name = "revenue_kept_php", nullable = false, precision = 14, scale = 2)
    var revenueKeptPhp: BigDecimal = BigDecimal.ZERO,

    @Column(name = "first_upload_at")
    var firstUploadAt: OffsetDateTime? = null,

    @Column(name = "last_upload_at")
    var lastUploadAt: OffsetDateTime? = null,
)
