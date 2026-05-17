package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.time.OffsetDateTime
import java.util.UUID

// Wire form lowercase snake_case so JSON pass-through stays direct. CHECK
// constraint in V10 (extended in V15) mirrors this list — adding a new
// kind needs both the enum and the constraint to update together.
enum class PhotographerMessageKind(val wire: String) {
    VERIFICATION_APPROVED("verification_approved"),
    VERIFICATION_REJECTED("verification_rejected"),
    VERIFICATION_RESET("verification_reset"),
    SUSPENDED("suspended"),
    UNSUSPENDED("unsuspended"),
    FORCE_EDIT("force_edit"),
    DISPUTE_RESOLVED("dispute_resolved"),
    DISPUTE_DENIED("dispute_denied"),
    DISPUTE_ESCALATED("dispute_escalated"),
    PAYOUT_APPROVED("payout_approved"),
    PAYOUT_HELD("payout_held"),
    PAYOUT_PAID("payout_paid"),
    PAYOUT_REPORT_ACKNOWLEDGED("payout_report_acknowledged"),
    PAYOUT_REPORT_RESOLVED("payout_report_resolved"),
    ADMIN_MESSAGE("admin_message");
}

// DB-backed photographer inbox per Q-A2. Admin actions write a row in the
// same TX as the decision-log entry; the photographer reads via
// GET /api/v1/me/photographer/messages (V15 wired the FE surface).
//
// `title` is populated by ADMIN_MESSAGE (admin-supplied subject) and is
// null for all other kinds — the FE derives a label from MESSAGE_KIND_LABEL.
// `removedAt` is the photographer's soft-delete flag; the list endpoint
// filters removed_at IS NULL.
@Entity
@Table(name = "photographer_messages")
class PhotographerMessage(
    @Id
    @Column(name = "id", nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "photographer_id", nullable = false, updatable = false)
    val photographerId: UUID,

    @Column(name = "kind", nullable = false, length = 40)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    val kindWire: String,

    @Column(name = "title", columnDefinition = "TEXT")
    val title: String? = null,

    @Column(name = "body", nullable = false, columnDefinition = "TEXT")
    val body: String,

    @Column(name = "source_admin_id")
    val sourceAdminId: UUID? = null,

    @Column(name = "source_decision_id")
    val sourceDecisionId: UUID? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "read_at")
    var readAt: OffsetDateTime? = null,

    @Column(name = "removed_at")
    var removedAt: OffsetDateTime? = null,
) {
    val kind: PhotographerMessageKind
        get() = PhotographerMessageKind.entries.first { it.wire == kindWire }
}
