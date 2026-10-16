package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

// Cycle id is a human-readable string PK (Q-A1) so receipts, share URLs, and
// report references can use it directly without a UUID ↔ display-id
// translation step. Shapes: `buildSequencedCycleId` (requests),
// `buildWeeklyCycleId` (weekly generator); older rows keep their ids.
// status/method are stored as the lowercase wire form (Photo.spanWire pattern)
// so DTO mapping is a direct pass-through.
@Entity
@Table(name = "payout_cycles")
class PayoutCycle(
    @Id
    @Column(name = "id", nullable = false, updatable = false, length = 80)
    val id: String,

    @Column(name = "photographer_id", nullable = false, updatable = false)
    val photographerId: UUID,

    @Column(name = "week_of", nullable = false)
    var weekOf: LocalDate,

    @Column(name = "amount_php", nullable = false, precision = 14, scale = 2)
    var amountPhp: BigDecimal = BigDecimal.ZERO,

    @Column(name = "item_count", nullable = false)
    var itemCount: Int = 0,

    @Column(name = "status", nullable = false, length = 20)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    var statusWire: String = PayoutCycleStatus.SCHEDULED.wire,

    @Column(name = "method", nullable = false, length = 20)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    var methodWire: String = PayoutMethod.GCASH.wire,

    @Column(name = "settled_at")
    var settledAt: OffsetDateTime? = null,

    @Column(name = "payment_reference", length = 100)
    var paymentReference: String? = null,

    @Column(name = "hold_reason", length = 255)
    var holdReason: String? = null,

    // ─── Explicit lifecycle (V56, Phase 4b) — set as the cycle moves through
    // the admin state machine. settledAt keeps its meaning as the paid-at
    // instant; approvedAt is the review instant. second_approved_by is Phase 4c.
    @Column(name = "approved_by")
    var approvedBy: UUID? = null,

    @Column(name = "approved_at")
    var approvedAt: OffsetDateTime? = null,

    @Column(name = "second_approved_by")
    var secondApprovedBy: UUID? = null,

    @Column(name = "processing_at")
    var processingAt: OffsetDateTime? = null,

    @Column(name = "paid_by")
    var paidBy: UUID? = null,

    @Column(name = "failed_reason", length = 255)
    var failedReason: String? = null,

    @Column(name = "returned_at")
    var returnedAt: OffsetDateTime? = null,

    @Column(name = "rejected_reason", length = 255)
    var rejectedReason: String? = null,

    // ─── Disbursement seam (V58, Phase 8) ─────────────────────────────────
    // Which rail moved the money and its id there. 'manual' = admin sends by
    // hand (today); 'paymongo' = a Transfer created via the API (Phase 11).
    @Column(name = "provider", nullable = false, length = 20)
    var provider: String = "manual",

    @Column(name = "provider_transfer_id", length = 100)
    var providerTransferId: String? = null,

    // Destination snapshot captured at request time — immutable, so a later
    // payout-account change can't reroute an in-flight payout. Disbursement
    // targets this, not the live primary account.
    @Column(name = "account_snapshot_method", length = 20)
    var accountSnapshotMethod: String? = null,

    @Column(name = "account_snapshot_number", length = 64)
    var accountSnapshotNumber: String? = null,

    @Column(name = "account_snapshot_name", length = 200)
    var accountSnapshotName: String? = null,

    @Column(name = "account_snapshot_qr_key", length = 512)
    var accountSnapshotQrKey: String? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),
) {
    var status: PayoutCycleStatus
        get() = PayoutCycleStatus.fromWire(statusWire)
        set(value) {
            statusWire = value.wire
        }

    var method: PayoutMethod
        get() = PayoutMethod.fromWire(methodWire)
            ?: throw IllegalStateException("Unknown payout method on cycle $id: $methodWire")
        set(value) {
            methodWire = value.wire
        }
}
