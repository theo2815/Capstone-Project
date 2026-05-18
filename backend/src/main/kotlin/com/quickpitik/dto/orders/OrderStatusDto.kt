package com.quickpitik.dto.orders

import com.quickpitik.entity.OrderStatus
import java.time.OffsetDateTime
import java.util.UUID

// Minimal status payload for the guest polling endpoint
// `GET /api/v1/orders/{id}/status?token=…`. Authed users use
// `GET /api/v1/me/orders/{id}` for the full hydrated detail.
//
// Only the fields the /orders/return polling page needs are exposed
// here — by design, no item/photo URLs leak through the guest path
// without a verified share token + (future) email link.
data class OrderStatusDto(
    val id: UUID,
    val status: OrderStatus,
    val paidAt: OffsetDateTime?,
)
