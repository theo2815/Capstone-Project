package com.quickpitik.entity

import jakarta.persistence.*
import org.hibernate.annotations.Immutable
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.time.Instant
import java.util.UUID

@Entity
@Immutable
@Table(name = "photo_removal_history")
class PhotoRemovalHistory(
    @Id @Column(nullable = false, updatable = false)
    val id: UUID,
    @Column(name = "request_id", nullable = false, updatable = false)
    val requestId: UUID,
    @Column(name = "item_id", updatable = false)
    val itemId: UUID?,
    @Column(name = "review_id", updatable = false)
    val reviewId: UUID?,
    @Column(name = "photo_id_snapshot", updatable = false)
    val photoIdSnapshot: UUID?,
    @Enumerated(EnumType.STRING) @Column(name = "actor_type", nullable = false, updatable = false, length = 16)
    val actorType: RemovalActorType,
    @Column(name = "actor_id_snapshot", updatable = false)
    val actorIdSnapshot: UUID?,
    @Column(nullable = false, updatable = false, length = 24)
    val action: String,
    @Column(name = "from_state", updatable = false, length = 24)
    val fromState: String?,
    @Column(name = "to_state", updatable = false, length = 24)
    val toState: String?,
    @Column(updatable = false, columnDefinition = "TEXT")
    val reason: String?,
    @Column(name = "occurred_at", nullable = false, updatable = false)
    val occurredAt: Instant,
    @Column(name = "command_id", nullable = false, updatable = false)
    val commandId: UUID,
    @JdbcTypeCode(SqlTypes.JSON) @Column(nullable = false, updatable = false, columnDefinition = "jsonb")
    val metadata: Map<String, String>,
)
