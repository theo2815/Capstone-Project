package com.quickpitik.mobile.data.remote

data class AddCartItemRequest(
    val photoId: String,
    val eventId: String
)

data class CartItemDto(
    val photoId: String,
    val eventId: String,
    val thumbnailUrl: String?,
    val price: Double,
    val bib: String?,
    val eventName: String?,
    val eventSlug: String?,
    val tone: Int?,
    val time: String?
)

data class MergeCartRequest(
    val items: List<MergeCartItem>
)

data class MergeCartItem(
    val photoId: String,
    val eventId: String
)

// Mirrors backend dto.cart.ClearedResponse — `cleared` is the COUNT of items
// removed (Int), not a Boolean. Used to be Boolean here, which Gson silently
// failed to parse against the int payload; the catch block then reverted the
// optimistic local clear and the cart kept its items after PayMongo paid.
data class ClearedResponse(
    val cleared: Int
)

data class RemovedResponse(
    val removed: Boolean
)

data class CreateOrderRequest(
    val items: List<CreateOrderItem>,
    val paymentMethod: String,
    val recipientEmail: String?,
    // Sent as "android" so the backend points PayMongo's success/cancel URLs
    // at the MobileReturnController bridge, which deep-links back into the app
    // via `quickpitik://` instead of stranding the user on localhost:3000.
    val clientPlatform: String = "android",
    // Photographer coupon (V45/V47). Validated + priced server-side only; a code
    // that covers none of the items is 400 COUPON_NOT_APPLICABLE.
    val couponCode: String? = null,
)

data class CreateOrderItem(
    val photoId: String,
    val eventId: String
)

data class OrderResponse(
    val id: String,
    val status: String, // PENDING | PAID | FULFILLED | REFUNDED | EXPIRED (UPPERCASE)
    val items: List<OrderResponseItem>,
    val totalAmount: Double,
    val paymentMethod: String,
    val createdAt: String,
    val redirectUrl: String? = null,
    val couponCode: String? = null,
    // ₱0 inline settlement (V46/V50 100% coupon) or a paid-key replay: the order
    // is already FULFILLED, redirectUrl is null, and this guest RETURN token loads
    // the receipt without a redirect. Null for signed-in runners and pending orders.
    val returnToken: String? = null,
)

data class OrderResponseItem(
    val photoId: String,
    val price: Double,
    val downloadUrl: String? = null
)

data class OrderListItemDto(
    val id: String,
    val eventId: String,
    val photoIds: List<String>,
    val total: Double,
    val paymentMethod: String,
    val paidAt: String?,
    val eventName: String?,
    val eventSlug: String?,
    val eventDate: String?,
    val eventState: String?,
    val status: String?,
    val disputes: List<RunnerDisputeDto> = emptyList()
)

data class OrderDetailDto(
    val id: String,
    val eventId: String,
    val photoIds: List<String>,
    val total: Double,
    val paymentMethod: String,
    val paidAt: String?,
    val eventName: String?,
    val eventSlug: String?,
    val status: String?,
    val photos: List<OrderPhotoDetailDto>,
    val downloadBundleUrl: String? = null,
    val recipientEmail: String = "",
    val shareToken: String? = null,
    val disputes: List<RunnerDisputeDto> = emptyList()
)

data class OrderPhotoDetailDto(
    val id: String,
    val bib: String?,
    val time: String,
    val tone: Int,
    val thumbnailUrl: String? = null,
    val previewUrl: String? = null,
    val downloadUrl: String? = null
)

// Mirrors backend RefundRequest — one Dispute row is created per photoId;
// reason + note are shared across the batch. reason is one of:
// "wrong_runner" | "low_quality" | "not_received" | "duplicate_charge" | "other".
data class RefundRequest(
    val photoIds: List<String>,
    val reason: String,
    val note: String = ""
)

data class RefundResponse(
    val orderId: String,
    val disputes: List<RefundDisputeDto> = emptyList()
)

data class RefundDisputeDto(
    val id: String,
    val photoId: String,
    val status: String,
    val openedAt: String
)

// Embedded on every order payload. Carries everything the runner-side refund
// surface needs to render status chips, the timeline, the admin's resolution
// note, and the cancel button. status is the lowercase wire string:
// "open" | "escalated" | "resolved" | "denied" | "withdrawn".
data class RunnerDisputeDto(
    val id: String,
    val photoId: String,
    val reason: String,
    val note: String,
    val status: String,
    val resolution: String? = null,
    val refundAmount: Double? = null,
    val resolutionNote: String? = null,
    val openedAt: String,
    val resolvedAt: String? = null,
    val withdrawnAt: String? = null
)

// POST /coupons/preview { photoIds, code? } — server prices coupon-eligible
// items so mobile never does money math. code optional: omitted = auto-applied
// photographer coupons; provided = validate + apply the typed override. Public
// (guest checkout allowed). Backend CouponDtos.
data class CouponPreviewRequest(
    val photoIds: List<String>,
    val code: String? = null,
)

data class CouponPreviewItemDto(
    val photoId: String,
    val price: Double,
    val discount: Double,
    val couponCode: String,
    val percentOff: Int,
)

// `items` holds discounted photos only; anything absent is full price. The
// top-level code fields describe the typed code (null when none was typed).
data class CouponPreviewDto(
    val code: String? = null,
    val percentOff: Int? = null,
    val photographerName: String? = null,
    val photographerHandle: String? = null,
    val items: List<CouponPreviewItemDto> = emptyList(),
    val eligibleCount: Int = 0,
    val discountTotal: Double = 0.0,
)

// GET /me/orders/{id}/status + GET /orders/{id}/status (V39 status-only poll).
// status: PENDING | PAID | FULFILLED | REFUNDED | EXPIRED. Backend OrderStatusDto.
data class OrderStatusDto(
    val id: String,
    val status: String,
    val paidAt: String? = null,
)
