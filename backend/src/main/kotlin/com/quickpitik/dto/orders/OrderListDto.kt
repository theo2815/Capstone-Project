package com.quickpitik.dto.orders

import com.quickpitik.entity.OrderStatus
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// GET /me/orders item — matches website MockOrder shape, including photoIds[]
// per the api-orders.ts ADR addendum (refund modal needs them without a detail
// round-trip).
data class OrderListItemDto(
    val id: UUID,
    val eventId: UUID,
    val photoIds: List<UUID>,
    val total: BigDecimal,
    val paymentMethod: String,
    val paidAt: OffsetDateTime?,
    val eventName: String?,
    val eventSlug: String?,
    val status: OrderStatus?,
)
