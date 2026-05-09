package com.quickpitik.dto.orders

import com.quickpitik.entity.OrderStatus
import jakarta.validation.constraints.Email
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotEmpty
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// POST /orders request body matches website/src/lib/api-orders.ts CreateOrderArgs.
// items[] may span multiple events; service splits server-side so each Order row
// stays single-event (keeps MockOrder.eventId honest).
data class CreateOrderRequest(
    @field:NotEmpty(message = "items must not be empty")
    val items: List<CreateOrderItem> = emptyList(),
    @field:NotBlank(message = "paymentMethod is required")
    val paymentMethod: String = "",
    @field:Email(message = "recipientEmail must be a valid email")
    val recipientEmail: String? = null,
    @field:NotBlank(message = "idempotencyKey is required")
    val idempotencyKey: String = "",
)

data class CreateOrderItem(
    val photoId: UUID,
    val eventId: UUID,
)

// POST /orders response — matches website/src/types/order.ts Order interface.
// Returned: the FIRST order created (smallest event date / first event_id) so the
// FE has a stable id to navigate to. Multi-event checkouts produce N rows; the
// remaining N-1 are visible via the next GET /me/orders fetch.
data class OrderResponse(
    val id: UUID,
    val status: OrderStatus,
    val items: List<OrderResponseItem>,
    val totalAmount: BigDecimal,
    val paymentMethod: String,
    val createdAt: OffsetDateTime,
    val redirectUrl: String? = null,
)

data class OrderResponseItem(
    val photoId: UUID,
    val price: BigDecimal,
    val downloadUrl: String? = null,
)
