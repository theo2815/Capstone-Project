package com.quickpitik.entity

import jakarta.persistence.*
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import com.fasterxml.jackson.databind.JsonNode
import java.time.Instant
import java.util.UUID

enum class RemovalCommandOperation { CREATE, REPLACE_EVIDENCE, CANCEL, MATCH_CANDIDATES, VERIFY }
enum class RemovalCommandState { PENDING, UNKNOWN, SUCCEEDED, FAILED }

@Entity
@Table(name = "photo_removal_commands")
class PhotoRemovalCommand(
    @Id @Column(name = "command_id", nullable = false, updatable = false)
    val commandId: UUID,
    @Column(name = "request_id", nullable = false, updatable = false)
    val requestId: UUID,
    @Enumerated(EnumType.STRING) @Column(nullable = false, updatable = false, length = 24)
    val operation: RemovalCommandOperation,
    @Column(name = "actor_id_snapshot", nullable = false, updatable = false)
    val actorIdSnapshot: UUID,
    @Column(name = "input_hash", nullable = false, updatable = false, length = 64)
    val inputHash: String,
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 16)
    var state: RemovalCommandState = RemovalCommandState.PENDING,
    @Column(name = "intended_evidence_key", updatable = false, length = 512)
    val intendedEvidenceKey: String? = null,
    @JdbcTypeCode(SqlTypes.JSON) @Column(columnDefinition = "jsonb")
    var result: JsonNode? = null,
    @Column(name = "claimed_at", nullable = false, updatable = false)
    val claimedAt: Instant,
)
