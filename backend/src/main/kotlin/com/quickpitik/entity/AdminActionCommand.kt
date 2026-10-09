package com.quickpitik.entity

import jakarta.persistence.*
import java.time.Instant
import java.util.UUID

enum class AdminActionOperation { USER_MESSAGE, EVENT_CREATE, EVENT_UPDATE }
enum class AdminActionCommandState { PENDING, UNKNOWN, SUCCEEDED, FAILED }

@Entity
@Table(name = "admin_action_commands")
class AdminActionCommand(
    @Id @Column(name = "command_id", nullable = false, updatable = false) val commandId: UUID,
    @Column(name = "actor_id", nullable = false, updatable = false) val actorId: UUID,
    @Enumerated(EnumType.STRING) @Column(nullable = false, updatable = false, length = 24) val operation: AdminActionOperation,
    @Column(name = "target_id", updatable = false) val targetId: UUID?,
    @Column(name = "input_hash", nullable = false, updatable = false, length = 64) val inputHash: String,
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 16) var state: AdminActionCommandState = AdminActionCommandState.PENDING,
    @Column(name = "claim_owner", nullable = false) var claimOwner: UUID = UUID.randomUUID(),
    @Column(name = "lease_expires_at") var leaseExpiresAt: Instant? = null,
    @Column(name = "intended_event_id", updatable = false) val intendedEventId: UUID? = null,
    @Column(name = "intended_storage_key", updatable = false, length = 512) val intendedStorageKey: String? = null,
    @Column(name = "processed_cover_digest", updatable = false, length = 64) val processedCoverDigest: String? = null,
    @Column(name = "result_resource_id") var resultResourceId: UUID? = null,
    @Column(name = "result_decision_id") var resultDecisionId: UUID? = null,
    @Column(name = "result_message_id") var resultMessageId: UUID? = null,
    @Column(name = "claimed_at", nullable = false, updatable = false) val claimedAt: Instant,
    @Column(name = "updated_at", nullable = false) var updatedAt: Instant,
    @Column(name = "completed_at") var completedAt: Instant? = null,
)
