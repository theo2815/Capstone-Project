package com.quickpitik.entity

import jakarta.persistence.*
import java.time.Instant
import java.util.UUID

@Entity
@Table(name = "photo_removal_reviews")
class PhotoRemovalReview(
    @Id @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),
    @Column(name = "request_id", nullable = false, updatable = false)
    val requestId: UUID,
    @Column(name = "owner_id_snapshot", nullable = false, updatable = false)
    val ownerIdSnapshot: UUID,
    @Column(name = "owner_id")
    var ownerId: UUID? = null,
    @Column(name = "notified_message_id")
    var notifiedMessageId: UUID? = null,
    @Column(name = "notified_at")
    var notifiedAt: Instant? = null,
    @Column(name = "deadline")
    var deadline: Instant? = null,
    @Column(nullable = false)
    var overdue: Boolean = false,
    @Column(name = "overdue_at")
    var overdueAt: Instant? = null,
    @Column(name = "unavailable_reason", columnDefinition = "TEXT")
    var unavailableReason: String? = null,
    @Column(name = "notification_next_attempt_at")
    var notificationNextAttemptAt: Instant? = null,
)
