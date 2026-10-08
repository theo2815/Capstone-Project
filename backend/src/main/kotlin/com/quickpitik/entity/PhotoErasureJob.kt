package com.quickpitik.entity

import com.quickpitik.dto.photos.PhotoErasureOutcome
import com.quickpitik.dto.photos.PhotoErasurePurpose
import jakarta.persistence.*
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.time.Instant
import java.util.UUID

data class ErasureIntent(val id: UUID, val stagingKey: String, val originalKey: String, val recheckUntil: Instant)
data class ErasureManifest(val keys: List<String>, val intents: List<ErasureIntent>, val neverIndexed: Boolean)
data class ErasureAuthorization(val actorId: UUID, val requestId: UUID, val itemId: UUID,
    val reason: String, val commandId: UUID, val occurredAt: Instant)

@Entity
@Table(name = "photo_erasure_jobs")
class PhotoErasureJob(
    @Id val id: UUID = UUID.randomUUID(),
    @Column(name = "photo_id", nullable = false, updatable = false) val photoId: UUID,
    @Column(name = "event_id", nullable = false, updatable = false) val eventId: UUID,
    @Column(name = "owner_id", updatable = false) val ownerId: UUID?,
    @Enumerated(EnumType.STRING) @Column(nullable = false, updatable = false) val purpose: PhotoErasurePurpose,
    @Enumerated(EnumType.STRING) @Column(name = "authorized_actor_type", nullable = false, updatable = false) val authorizedActorType: RemovalActorType,
    @Column(name = "authorized_actor_id", nullable = false, updatable = false) val authorizedActorId: UUID,
    @Column(nullable = false, updatable = false) val reason: String,
    @Column(name = "request_id", updatable = false) val requestId: UUID?,
    @Column(name = "item_id", updatable = false) val itemId: UUID?,
    @Column(name = "command_id", nullable = false, updatable = false) val commandId: UUID,
    @JdbcTypeCode(SqlTypes.JSON) @Column(nullable = false, updatable = false, columnDefinition = "jsonb") val manifest: ErasureManifest,
    @JdbcTypeCode(SqlTypes.JSON) @Column(name = "additional_intents", nullable = false, columnDefinition = "jsonb") var additionalIntents: List<ErasureIntent> = emptyList(),
    @Column(nullable = false, updatable = false) val generation: Long,
    @JdbcTypeCode(SqlTypes.JSON) @Column(nullable = false, columnDefinition = "jsonb") var authorizations: List<ErasureAuthorization> = emptyList(),
    @Column(name = "retain_tombstone", nullable = false) var retainTombstone: Boolean = purpose == PhotoErasurePurpose.PRIVACY_REMOVAL,
    @JdbcTypeCode(SqlTypes.JSON) @Column(name = "resource_outcomes", nullable = false, columnDefinition = "jsonb") var resourceOutcomes: Map<String, String> = emptyMap(),
    @Column(name = "face_outcome", nullable = false) var faceOutcome: String = "PENDING",
    @Enumerated(EnumType.STRING) @Column(nullable = false) var outcome: PhotoErasureOutcome = PhotoErasureOutcome.PENDING,
    @Column(nullable = false) var revision: Long = 0,
    @Column(nullable = false) var attempts: Int = 0,
    @Column(name = "lease_token") var leaseToken: UUID? = null,
    @Column(name = "lease_until") var leaseUntil: Instant? = null,
    @Column(name = "next_attempt_at", nullable = false) var nextAttemptAt: Instant,
    @Column(name = "last_error") var lastError: String? = null,
    @Enumerated(EnumType.STRING) @Column(name = "execution_actor_type") var executionActorType: RemovalActorType? = null,
    @Column(name = "execution_actor_id") var executionActorId: UUID? = null,
    @Column(name = "created_at", nullable = false, updatable = false) val createdAt: Instant,
    @Column(name = "updated_at", nullable = false) var updatedAt: Instant = createdAt,
    @Column(name = "completed_at") var completedAt: Instant? = null,
)
