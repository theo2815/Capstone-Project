package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.time.OffsetDateTime
import java.util.UUID

enum class FlagStatus(val wire: String) {
    OPEN("open"), RESOLVED("resolved"), HIDDEN("hidden"), DISMISSED("dismissed"), ESCALATED("escalated");

    companion object {
        fun fromWire(value: String): FlagStatus =
            entries.firstOrNull { it.wire == value.trim().lowercase() }
                ?: throw IllegalArgumentException("Unknown flag status: $value")
    }
}

enum class FlagTargetKind(val wire: String) {
    PHOTO("photo"), USER("user"), EVENT("event");

    companion object {
        fun fromWire(value: String): FlagTargetKind =
            entries.firstOrNull { it.wire == value.trim().lowercase() }
                ?: throw IllegalArgumentException("Unknown flag target kind: $value")
    }
}

// Minimal flagging surface. Phase G ships the env-gated read + state-transition
// endpoints so the queue exists; richer triage workflow (assignee, severity,
// auto-cull thresholds) lands when v1 actually starts ingesting flags
// (Q-A5 follow-up).
@Entity
@Table(name = "flags")
class Flag(
    @Id
    @Column(name = "id", nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "target_kind", nullable = false, length = 20)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    var targetKindWire: String,

    @Column(name = "target_id", nullable = false)
    var targetId: UUID,

    @Column(name = "reporter_id")
    var reporterId: UUID? = null,

    @Column(name = "reason", nullable = false, length = 40)
    var reason: String,

    @Column(name = "note", nullable = false, columnDefinition = "TEXT")
    var note: String = "",

    @Column(name = "status", nullable = false, length = 20)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    var statusWire: String = FlagStatus.OPEN.wire,

    @Column(name = "resolution_note", columnDefinition = "TEXT")
    var resolutionNote: String? = null,

    @Column(name = "resolved_by")
    var resolvedBy: UUID? = null,

    @Column(name = "resolved_at")
    var resolvedAt: OffsetDateTime? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),
) {
    var status: FlagStatus
        get() = FlagStatus.fromWire(statusWire)
        set(value) { statusWire = value.wire }

    var targetKind: FlagTargetKind
        get() = FlagTargetKind.fromWire(targetKindWire)
        set(value) { targetKindWire = value.wire }
}
