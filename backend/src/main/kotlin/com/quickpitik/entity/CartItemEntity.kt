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
     * The price the runner was last shown for this photo. **Not** the charged
     * price: `OrderService.create` always charges the live `photos.price_php`.
     * Uses: `CartService.add` raises CART_ITEM_PRICE_CHANGED (H-4) on a re-add
     * of a re-priced photo; the cart shows "went up from ₱X" while the live
     * price is higher (AA-R14); a cart read lowers it to a lower live price
     * (AF-18 F5); checkout refuses a rise once when the client sends no
     * expected price, then raises it (AF-12, `OrderService.requireExpectedPrices`).
     */
    @Column(name = "price_php_at_add", nullable = false, precision = 12, scale = 2)
    var pricePhpAtAdd: BigDecimal,

    @Column(name = "added_at", nullable = false)
    var addedAt: OffsetDateTime = OffsetDateTime.now(),
)
