package com.quickpitik.entity

import jakarta.persistence.*
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.time.Instant
import java.util.UUID

@Entity
@Table(name = "photo_removal_requests")
class PhotoRemovalRequest(
    @Id @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),
    @Column(name = "event_id")
    var eventId: UUID? = null,
    @Column(name = "event_id_snapshot", nullable = false, updatable = false)
    val eventIdSnapshot: UUID,
    @Column(name = "event_name_snapshot", nullable = false, updatable = false)
    val eventNameSnapshot: String,
    @Column(name = "requester_user_id")
    var requesterUserId: UUID? = null,
    @Column(name = "requester_contact", nullable = false, columnDefinition = "TEXT")
    var requesterContact: String,
    @Column(name = "support_reference", nullable = false, columnDefinition = "TEXT")
    var supportReference: String,
    @Column(name = "requester_reason", nullable = false, columnDefinition = "TEXT")
    var requesterReason: String,
    @Column(name = "public_to_owner_reason", nullable = false, columnDefinition = "TEXT")
    var publicToOwnerReason: String,
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 16)
    var status: RemovalRequestStatus = RemovalRequestStatus.DRAFT,
    @Version @Column(nullable = false)
    var version: Long = 0,
    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: Instant = Instant.now(),
    @Column(name = "updated_at", nullable = false)
    var updatedAt: Instant = createdAt,
    @Column(name = "resolved_at")
    var resolvedAt: Instant? = null,
    @Column(name = "creator_id", nullable = false, updatable = false)
    val creatorId: UUID,
    @Column(name = "verifier_id")
    var verifierId: UUID? = null,
    @Column(name = "verification_method", columnDefinition = "TEXT")
    var verificationMethod: String? = null,
    @Column(name = "verification_note", columnDefinition = "TEXT")
    var verificationNote: String? = null,
    @Column(name = "verified_at")
    var verifiedAt: Instant? = null,
    @Column(name = "evidence_key", length = 512)
    var evidenceKey: String? = null,
    @Column(name = "evidence_content_type", length = 100)
    var evidenceContentType: String? = null,
    @Column(name = "evidence_generation", nullable = false)
    var evidenceGeneration: Long = 0,
    @Enumerated(EnumType.STRING) @Column(name = "evidence_cleanup_state", nullable = false, length = 16)
    var evidenceCleanupState: RemovalEvidenceCleanupState = RemovalEvidenceCleanupState.NONE,
    @JdbcTypeCode(SqlTypes.JSON) @Column(name = "outstanding_evidence_keys", nullable = false, columnDefinition = "jsonb")
    var outstandingEvidenceKeys: List<String> = emptyList(),
    @Column(name = "creation_command_id", nullable = false, updatable = false)
    val creationCommandId: UUID,
)
