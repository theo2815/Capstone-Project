package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.Table
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

@Entity
@Table(name = "orders")
class Order(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "user_id")
    var userId: UUID? = null,

    @Column(name = "event_id", nullable = false)
    var eventId: UUID,

    @Column(name = "recipient_email", nullable = false, length = 254)
    var recipientEmail: String,

    @Column(name = "payment_method", nullable = false, length = 20)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    var paymentMethodWire: String,

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 20)
    var status: OrderStatus = OrderStatus.PENDING,

    @Column(name = "total_php", nullable = false, precision = 12, scale = 2)
    var totalPhp: BigDecimal,

    @Column(name = "idempotency_key", length = 64)
    var idempotencyKey: String? = null,

    @Column(name = "paid_at")
    var paidAt: OffsetDateTime? = null,

    // Opaque token minted for guest orders so /orders/{id}?token=... is
    // reachable from the email receipt without an account. NULL for authed
    // orders — those reach the page via JWT.
    @Column(name = "share_token", length = 64, unique = true)
    var shareToken: String? = null,

    // Stamped after the Resend receipt fires. Idempotency guard against
    // PayMongo's retried webhook deliveries — handler checks NULL before send.
    @Column(name = "email_sent_at")
    var emailSentAt: OffsetDateTime? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),
) {
    var paymentMethod: PaymentMethod
        get() = PaymentMethod.fromWire(paymentMethodWire)
        set(value) { paymentMethodWire = value.wire }
}
