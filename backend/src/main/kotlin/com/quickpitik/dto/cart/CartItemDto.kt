package com.quickpitik.dto.cart

import java.math.BigDecimal
import java.util.UUID

// Mirrors website/src/types/order.ts CartItem.
data class CartItemDto(
    val photoId: UUID,
    val eventId: UUID,
    val thumbnailUrl: String,
    val price: BigDecimal,
    val bib: String? = null,
    val eventName: String? = null,
    val eventSlug: String? = null,
    val tone: Int? = null,
    val time: String? = null,
    /** False only for a suspended photographer's photo, and only on `GET /me/cart?includeUnavailable=true`. */
    val available: Boolean = true,
    /** The price when the photo was added, or a lower price the cart showed since (AF-18 F5); clients show "went up from ₱priceAtAdd" while `price` is higher (AA-R14). */
    val priceAtAdd: BigDecimal,
)

data class AddCartItemRequest(
    val photoId: UUID,
    val eventId: UUID,
)

data class MergeCartRequest(
    val items: List<MergeCartItem> = emptyList(),
)

data class MergeCartItem(
    val photoId: UUID,
    val eventId: UUID,
    /** The guest cart's add-time price; kept only when 0 < value ≤ the live price, else the live price (AA-R14). */
    val priceAtAdd: BigDecimal? = null,
)
