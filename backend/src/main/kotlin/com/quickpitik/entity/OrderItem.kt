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
)
