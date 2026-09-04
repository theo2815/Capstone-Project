package com.quickpitik.dto.orders

import com.quickpitik.entity.OrderStatus
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

// GET /me/orders item — matches website MockOrder shape, including photoIds[]
// per the api-orders.ts ADR addendum (refund modal needs them without a detail
// round-trip). disputes[] embedded so the receipt status chip + cancel
// button + timeline can render off the list payload without a detail fetch.
// eventDate + eventState added 2026-05-19 PM for the /profile race log:
// purchased-only rows (events the runner bought but didn't save) need the
// date column + state label without an extra round-trip.
data class OrderListItemDto(
    val id: UUID,
    val eventId: UUID,
    val photoIds: List<UUID>,
    val total: BigDecimal,
    val paymentMethod: String,
    val paidAt: OffsetDateTime?,
    val eventName: String?,
    val eventSlug: String?,
    val eventDate: LocalDate?,
    val eventState: String?,
    val status: OrderStatus?,
    val disputes: List<RunnerDisputeDto> = emptyList(),
    // Photographer coupon (V45) — see OrderDetailDto.
    val couponCode: String? = null,
    val discountTotal: BigDecimal = BigDecimal.ZERO,
)
