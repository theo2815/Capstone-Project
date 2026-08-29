package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

enum class PaymentStatus { PENDING, SUCCEEDED, FAILED }

@Entity
@Table(name = "payments")
class Payment(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "order_id", nullable = false)
    var orderId: UUID,

    @Column(name = "provider", nullable = false, length = 20)
    var provider: String,

    @Column(name = "provider_ref", length = 100)
    var providerRef: String? = null,

    // Actual pay_... resource created by the Checkout Session. Refunds target
    // this ID, not the cs_... session stored in providerRef.
    @Column(name = "provider_payment_id", length = 100)
    var providerPaymentId: String? = null,

    @Column(name = "amount_php", nullable = false, precision = 12, scale = 2)
    var amountPhp: BigDecimal,

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 20)
    var status: PaymentStatus = PaymentStatus.PENDING,

    @Column(name = "paid_at")
    var paidAt: OffsetDateTime? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),
)
