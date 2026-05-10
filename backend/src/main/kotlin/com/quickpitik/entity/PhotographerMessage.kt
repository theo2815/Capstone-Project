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
// constraint in V10 mirrors this list — adding a new kind needs both the
// enum and the constraint to update together.
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
    PAYOUT_REPORT_RESOLVED("payout_report_resolved");
}

// DB-backed photographer inbox per Q-A2. Admin actions write a row in the
// same TX as the decision-log entry; the photographer surface polls
// /me/photographer/inbox (path TBD, deferred from PR 7 since the FE didn't
// surface it yet). PR 10 just writes the rows so the data is ready when
// that surface ships.
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
) {
    val kind: PhotographerMessageKind
        get() = PhotographerMessageKind.entries.first { it.wire == kindWire }
}
