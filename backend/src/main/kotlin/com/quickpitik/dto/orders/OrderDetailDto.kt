package com.quickpitik.dto.orders

import com.quickpitik.entity.OrderStatus
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// GET /me/orders/{id} — extends OrderListItemDto with photos[] + downloadBundleUrl.
// Mirrors website/src/lib/api-orders.ts OrderDetail (= MockOrder & { photos, downloadBundleUrl }).
data class OrderDetailDto(
    val id: UUID,
    val eventId: UUID,
    val photoIds: List<UUID>,
    val total: BigDecimal,
    val paymentMethod: String,
    val paidAt: OffsetDateTime?,
    val eventName: String?,
    val eventSlug: String?,
    val status: OrderStatus?,
    val photos: List<OrderPhotoDetailDto>,
    val downloadBundleUrl: String? = null,
    // Required by the /orders/return success state so we can render "We've
    // also emailed these links to <recipient>". Authed runners see their
    // account email; guests see whatever they typed at checkout.
    val recipientEmail: String = "",
    // Opaque per-order token. Both runners and guests use it to authorize the
    // bundle-download endpoint (top-level navigations can't carry the JWT).
    // Null only on legacy rows created before V18 backfilled them.
    val shareToken: String? = null,
    // Refund disputes filed against this order. Surfaces status, admin's
    // resolution note, and timestamps so /orders can render the chip +
    // timeline + cancel button off a single detail fetch.
    val disputes: List<RunnerDisputeDto> = emptyList(),
)

data class OrderPhotoDetailDto(
    val id: UUID,
    val bib: String?,
    val time: String,
    val tone: Int,
    val thumbnailUrl: String? = null,
    val previewUrl: String? = null,
    val downloadUrl: String? = null,
)
