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
)
