package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Embeddable
import jakarta.persistence.EmbeddedId
import jakarta.persistence.Entity
import jakarta.persistence.Table
import java.io.Serializable
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

@Embeddable
data class CartItemId(
    @Column(name = "user_id", nullable = false)
    var userId: UUID,

    @Column(name = "photo_id", nullable = false)
    var photoId: UUID,
) : Serializable

@Entity
@Table(name = "cart_items")
class CartItemEntity(
    @EmbeddedId
    val id: CartItemId,

    @Column(name = "event_id", nullable = false)
    var eventId: UUID,

    /**
     * The photo's price when this row was written. **Not** the displayed or
     * charged price — `CartService.toDto` renders and `OrderService.create`
     * charges the live `photos.price_php`. Its one live use is
     * `CartService.add`, which compares against it to raise
     * CART_ITEM_PRICE_CHANGED (H-4) when a runner re-adds a re-priced photo.
     */
    @Column(name = "price_php_at_add", nullable = false, precision = 12, scale = 2)
    var pricePhpAtAdd: BigDecimal,

    @Column(name = "added_at", nullable = false)
    var addedAt: OffsetDateTime = OffsetDateTime.now(),
)
