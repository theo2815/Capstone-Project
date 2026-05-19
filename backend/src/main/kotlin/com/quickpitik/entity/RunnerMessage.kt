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
// constraint in V20 mirrors this list — adding a new kind needs both the
// enum and the constraint to update together.
enum class RunnerMessageKind(val wire: String) {
    DISPUTE_RESOLVED("dispute_resolved"),
    DISPUTE_DENIED("dispute_denied"),
    DISPUTE_ESCALATED("dispute_escalated"),
    ADMIN_MESSAGE("admin_message"),
    ACCOUNT_SUSPENDED("account_suspended"),
    ACCOUNT_UNSUSPENDED("account_unsuspended");
}

// DB-backed runner inbox — parallel to PhotographerMessage. Admin actions
// targeting a runner (dispute outcomes, suspension, DM) write here in the
// same TX as the decision-log entry; the runner reads via
// GET /api/v1/me/runner/messages and is notified live via the
// /ws/me/runner/notifications WS channel.
//
// `orderId` deep-links dispute-outcome notifications to /orders?expand=…
// so a single click takes the runner to the affected receipt.
// `title` is populated by ADMIN_MESSAGE (admin-supplied subject) and is
// null for all other kinds.
// `removedAt` is the runner's soft-delete flag; the list endpoint filters
// removed_at IS NULL.
@Entity
@Table(name = "runner_messages")
class RunnerMessage(
    @Id
    @Column(name = "id", nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "runner_id", nullable = false, updatable = false)
    val runnerId: UUID,

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

    @Column(name = "order_id")
    val orderId: UUID? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "read_at")
    var readAt: OffsetDateTime? = null,

    @Column(name = "removed_at")
    var removedAt: OffsetDateTime? = null,
) {
    val kind: RunnerMessageKind
        get() = RunnerMessageKind.entries.first { it.wire == kindWire }
}
