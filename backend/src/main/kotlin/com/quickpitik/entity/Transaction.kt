package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// Per Q-E3 a refund is a separate row with negative amount_kept_php and
// is_refund=true, refund_of pointing to the originating sale. SUM(amount_kept_php)
// over a window then gives net earnings without a parallel ledger.
@Entity
@Table(name = "transactions")
class Transaction(
    @Id
    @Column(name = "id", nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    // Snapshot of the originating order's paid_at — keeps weekly/monthly
    // rollups stable even when this row is inserted retroactively.
    @Column(name = "paid_at", nullable = false)
    val paidAt: OffsetDateTime,

    @Column(name = "photographer_id", nullable = false, updatable = false)
    val photographerId: UUID,

    @Column(name = "event_id", nullable = false, updatable = false)
    val eventId: UUID,

    @Column(name = "photo_id", nullable = false, updatable = false)
    val photoId: UUID,

    @Column(name = "order_id", nullable = false, updatable = false)
    val orderId: UUID,

    @Column(name = "buyer_id")
    val buyerId: UUID? = null,

    @Column(name = "buyer_display_name", nullable = false, length = 100)
    val buyerDisplayName: String = "",

    @Column(name = "amount_kept_php", nullable = false, precision = 12, scale = 2)
    val amountKeptPhp: BigDecimal,

    @Column(name = "is_refund", nullable = false)
    val isRefund: Boolean = false,

    @Column(name = "refund_of")
    val refundOf: UUID? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),
)
