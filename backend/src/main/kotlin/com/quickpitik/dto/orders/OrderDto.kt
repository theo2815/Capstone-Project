package com.quickpitik.dto.orders

import com.quickpitik.entity.OrderStatus
import jakarta.validation.constraints.Email
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotEmpty
import jakarta.validation.constraints.Size
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// POST /orders request body matches website/src/lib/api-orders.ts CreateOrderArgs.
// items[] may span multiple events; service splits server-side so each Order row
// stays single-event (keeps MockOrder.eventId honest). Idempotency-Key arrives
// via the HTTP header (RFC 9110 §9.2.2), not the body — see OrderController.create.
data class CreateOrderRequest(
    @field:NotEmpty(message = "items must not be empty")
    val items: List<CreateOrderItem> = emptyList(),
    @field:NotBlank(message = "paymentMethod is required")
    val paymentMethod: String = "",
    @field:Email(message = "recipientEmail must be a valid email")
    val recipientEmail: String? = null,
    // When "android", OrderService picks the MobileReturnController bridge URLs
    // for PayMongo success/cancel so the user lands back in the app via the
    // quickpitik:// deep link instead of the website. null/empty → website flow.
    val clientPlatform: String? = null,
    // Photographer coupon (V45). Validated and priced server-side only; a code
    // that covers none of the items is a 400 COUPON_NOT_APPLICABLE.
    @field:Size(max = 32, message = "couponCode is too long")
    val couponCode: String? = null,
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
    val couponCode: String? = null,
)

// `price` is the list price; `discount` is the coupon's share of it (0 when
// none). totalAmount on the parent is what was actually charged.
data class OrderResponseItem(
    val photoId: UUID,
    val price: BigDecimal,
    val downloadUrl: String? = null,
    val discount: BigDecimal = BigDecimal.ZERO,
)
