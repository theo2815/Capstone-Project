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

// Cycle id is the human-readable `PAY-{YYYY}W{NN}-{HANDLE}` per Q-A1 — kept as
// the PK so receipts, share URLs, and report references can use it directly
// without a UUID ↔ display-id translation step. status/method are stored as
// the lowercase wire form (Photo.spanWire pattern) so DTO mapping is a direct
// pass-through.
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
