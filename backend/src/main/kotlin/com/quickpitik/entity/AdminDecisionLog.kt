package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.time.OffsetDateTime
import java.util.UUID

// Decision code for a refund PayMongo refused and an admin now settles by hand. It writes no
// money, so runner-facing dispute notes skip it (see OrderService / RefundService).
const val REFUND_SENT_TO_MANUAL_REVIEW = "refund_sent_to_manual_review"

// An admin saved a PayMongo refund link on a parked QR Ph dispute and told the runner. Writes no money either.
const val REFUND_LINK_SENT = "refund_link_sent"

// Append-only audit trail of every ADMIN write action. group_id is nullable
// per Q-A4 — bulk endpoints (POST /admin/payouts/bulk) share one UUID across
// every per-target row so the dashboard kpi `decisionsThisWeek` count can
// collapse a 50-payout bulk-hold down to one logical decision via
// `COUNT(DISTINCT COALESCE(group_id, id::text))`.
//
// Each decision targets exactly one of user / payout / dispute / event; the
// untouched columns stay NULL. `meta` carries decision-shape-specific JSON
// (e.g. force-edit = {field, from, to}; mark-paid = {paymentReference}; etc.)
// — keeping the column generic here means new decision shapes don't need
// schema migrations.
//
// idempotency_key is opt-in: bulk endpoints stamp the inbound Idempotency-Key
// header on every per-cycle row so a retry of the same admin intent reuses
// the cached group_id (and the partial unique on
// (admin_id, idempotency_key, target_payout_id) gates per-row dedup).
@Entity
@Table(name = "admin_decision_log")
class AdminDecisionLog(
    @Id
    @Column(name = "id", nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "admin_id", nullable = false, updatable = false)
    val adminId: UUID,

    @Column(name = "target_user_id")
    val targetUserId: UUID? = null,

    @Column(name = "target_payout_id", length = 80)
    val targetPayoutId: String? = null,

    @Column(name = "target_dispute_id")
    val targetDisputeId: UUID? = null,

    @Column(name = "target_event_id")
    val targetEventId: UUID? = null,

    @Column(name = "decision", nullable = false, length = 40)
    val decision: String,

    @Column(name = "reason", columnDefinition = "TEXT")
    val reason: String? = null,

    @Column(name = "meta", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    val meta: Map<String, Any?>? = null,

    @Column(name = "group_id")
    val groupId: UUID? = null,

    @Column(name = "idempotency_key", length = 64)
    val idempotencyKey: String? = null,

    @Column(name = "decided_at", nullable = false, updatable = false)
    val decidedAt: OffsetDateTime = OffsetDateTime.now(),
)
