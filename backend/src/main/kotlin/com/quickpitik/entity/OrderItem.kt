package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Embeddable
import jakarta.persistence.EmbeddedId
import jakarta.persistence.Entity
import jakarta.persistence.Table
import java.io.Serializable
import java.math.BigDecimal
import java.util.UUID

@Embeddable
data class OrderItemId(
    @Column(name = "order_id", nullable = false)
    var orderId: UUID,

    @Column(name = "photo_id", nullable = false)
    var photoId: UUID,
) : Serializable

@Entity
@Table(name = "order_items")
class OrderItem(
    @EmbeddedId
    val id: OrderItemId,

    @Column(name = "price_php_at_purchase", nullable = false, precision = 12, scale = 2)
    var pricePhpAtPurchase: BigDecimal,

    // Photographer-coupon discount on this item (V45); 0 when none applied.
    // pricePhpAtPurchase stays the list price — what the runner paid is
    // pricePhpAtPurchase − discountPhp, and that is what a refund returns.
    @Column(name = "discount_php", nullable = false, precision = 12, scale = 2)
    var discountPhp: BigDecimal = BigDecimal.ZERO,

    // The coupon that produced discountPhp (V50). Per item because an event
    // order can hold photos from several photographers, each with their own
    // coupon; usage limits count distinct orders through this column.
    @Column(name = "coupon_id")
    var couponId: UUID? = null,
)
