package com.quickpitik.entity

import jakarta.persistence.*
import java.time.Instant
import java.util.UUID

@Entity
@Table(name = "photo_removal_items")
class PhotoRemovalItem(
    @Id @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),
    @Column(name = "request_id", nullable = false, updatable = false)
    val requestId: UUID,
    @Column(name = "photo_id_snapshot", nullable = false, updatable = false)
    val photoIdSnapshot: UUID,
    @Column(name = "photo_id")
    var photoId: UUID? = null,
    @Column(name = "owner_id_snapshot", updatable = false)
    val ownerIdSnapshot: UUID? = null,
    @Column(name = "owner_name_snapshot", updatable = false)
    val ownerNameSnapshot: String? = null,
    @Column(name = "review_id")
    var reviewId: UUID? = null,
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 24)
    var state: RemovalItemState = RemovalItemState.CANDIDATE,
    @Enumerated(EnumType.STRING) @Column(name = "match_source", nullable = false, length = 10)
    val matchSource: RemovalMatchSource,
    @Column(name = "match_score")
    val matchScore: Double? = null,
    @Column(name = "verification_actor_id")
    var verificationActorId: UUID? = null,
    @Column(name = "verified_at")
    var verifiedAt: Instant? = null,
    @Column(name = "verification_reason", columnDefinition = "TEXT")
    var verificationReason: String? = null,
    @Enumerated(EnumType.STRING) @Column(name = "owner_response", length = 20)
    var ownerResponse: PhotographerRemovalResponse? = null,
    @Column(name = "owner_response_reason", columnDefinition = "TEXT")
    var ownerResponseReason: String? = null,
    @Column(name = "owner_responded_at")
    var ownerRespondedAt: Instant? = null,
    @Enumerated(EnumType.STRING) @Column(name = "admin_decision", length = 20)
    var adminDecision: RemovalDecision? = null,
    @Column(name = "admin_decision_reason", columnDefinition = "TEXT")
    var adminDecisionReason: String? = null,
    @Column(name = "admin_decision_actor_id")
    var adminDecisionActorId: UUID? = null,
    @Column(name = "admin_decided_at")
    var adminDecidedAt: Instant? = null,
    @Column(name = "retention_review_at")
    var retentionReviewAt: Instant? = null,
    @Column(name = "cleanup_job_id")
    var cleanupJobId: UUID? = null,
)
