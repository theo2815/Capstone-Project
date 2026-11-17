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

// Reasons the backend itself (or a signed-in user's report) writes through
// SystemFlagService; stored lowercase in flags.reason (VARCHAR(40), no CHECK).
enum class SystemFlagReason {
    PROFILE_IMAGE_FLAGGED, PROFILE_IMAGE_UNCHECKED, PAYOUT_NAME_CHANGED, REPEAT_ACCOUNT_MATCH,
    RACE_CHECK_MISS, RACE_EDIT_HELD, POSSIBLE_DUPLICATE, SAME_BIB_DUPLICATE,
    NO_PROOF_HIDDEN, NO_PROOF_UNINDEXED, REPORT_THRESHOLD_HIDDEN, USER_REPORT,
    RUNNER_REMOVAL_HELD, // RD-6: a runner request whose auto-hide was held by a cap
    // PE-2.1: one USER flag per photographer whose upload batch scored Low (relevance flag mode, retired in
    // RV-1.10: no new ones are raised; open ones stay in the queue until closed).
    BATCH_RELEVANCE_LOW;

    val wire: String get() = name.lowercase()
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

    // PE-2.2: an admin's payout hold on a relevance flag; it ends when the flag closes.
    @Column(name = "payout_hold_at")
    var payoutHoldAt: OffsetDateTime? = null,

    @Column(name = "payout_hold_by")
    var payoutHoldBy: UUID? = null,
) {
    var status: FlagStatus
        get() = FlagStatus.fromWire(statusWire)
        set(value) { statusWire = value.wire }

    var targetKind: FlagTargetKind
        get() = FlagTargetKind.fromWire(targetKindWire)
        set(value) { targetKindWire = value.wire }
}
