package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// "Notify me when my photos are ready" opt-in — one row per (event, runner).
// notification_claimed_at is the recoverable send lease; notified_at records
// only successful delivery.
@Entity
@Table(name = "event_photo_alerts")
class EventPhotoAlert(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "event_id", nullable = false)
    var eventId: UUID,

    @Column(name = "user_id", nullable = false)
    var userId: UUID,

    // The selfie the runner chose to be matched with. Null after they delete it
    // (FK ON DELETE SET NULL) — the notifier then falls back to primary/latest.
    @Column(name = "selfie_id")
    var selfieId: UUID? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "last_checked_at")
    var lastCheckedAt: OffsetDateTime? = null,

    // Written only through token-guarded repository updates. Excluding both
    // fields from managed-entity flushes prevents lastCheckedAt/selfie changes
    // from restoring stale notification state.
    @Column(name = "notified_at", updatable = false)
    var notifiedAt: OffsetDateTime? = null,

    @Column(name = "notification_claimed_at", updatable = false)
    var notificationClaimedAt: OffsetDateTime? = null,
)
